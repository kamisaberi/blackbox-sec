/**
 * @file cmaes_optimizer.cpp
 * @brief Covariance Matrix Adaptation Evolution Strategy (CMA-ES) Implementation
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#include <blackbox/optimizers/cmaes_optimizer.hpp>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <format>

namespace blackbox::optimizers {

CMAESOptimizer::CMAESOptimizer(const CMAESConfig& config)
    : m_config(config),
      m_rng(std::random_device{}()) {
    
    m_state.d = m_config.dimension;
    
    // Auto-calculate population size (lambda) if not specified: lambda = 4 + floor(3 * ln(d))
    if (m_config.population_size == 0) {
        m_state.lambda = static_cast<size_t>(4.0 + std::floor(3.0 * std::log(static_cast<double>(m_state.d))));
    } else {
        m_state.lambda = m_config.population_size;
    }

    m_state.sigma = m_config.initial_sigma;
    init_state();
}

void CMAESOptimizer::init_state() {
    size_t d = m_state.d;
    size_t lambda = m_state.lambda;

    // Parent population size (mu = lambda / 2)
    m_state.mu = lambda / 2;
    size_t mu = m_state.mu;

    // Compute logarithmic recombination weights
    m_state.weights.resize(mu);
    double sum_w = 0.0;
    for (size_t i = 0; i < mu; ++i) {
        m_state.weights[i] = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i + 1));
        sum_w += m_state.weights[i];
    }
    // Normalize weights
    for (size_t i = 0; i < mu; ++i) {
        m_state.weights[i] /= sum_w;
    }

    // Variance effective selection mass (mueff)
    double sum_w_sq = 0.0;
    for (size_t i = 0; i < mu; ++i) {
        sum_w_sq += m_state.weights[i] * m_state.weights[i];
    }
    m_state.mueff = 1.0 / sum_w_sq;

    double d_dbl = static_cast<double>(d);
    double mueff = m_state.mueff;

    // Strategy adaptation parameters
    m_state.cc = (4.0 + mueff / d_dbl) / (d_dbl + 4.0 + 2.0 * mueff / d_dbl);
    m_state.cs = (mueff + 2.0) / (d_dbl + mueff + 5.0);
    m_state.c1 = 2.0 / ((d_dbl + 1.3) * (d_dbl + 1.3) + mueff);
    m_state.cmu = std::min(1.0 - m_state.c1, 
        2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d_dbl + 2.0) * (d_dbl + 2.0) + mueff));
    m_state.damps = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff - 1.0) / (d_dbl + 1.0)) - 1.0) + m_state.cs;

    // Initialize mean vector m
    if (!m_config.initial_mean.empty() && m_config.initial_mean.size() == d) {
        m_state.m = m_config.initial_mean;
    } else {
        m_state.m.assign(d, 0.0);
    }

    // Initialize evolution paths
    m_state.pc.assign(d, 0.0);
    m_state.ps.assign(d, 0.0);

    // Initialize matrices (C = Identity, B = Identity, D = ones)
    m_state.C.assign(d * d, 0.0);
    m_state.B.assign(d * d, 0.0);
    m_state.D.assign(d, 1.0);

    for (size_t i = 0; i < d; ++i) {
        m_state.C[i * d + i] = 1.0;
        m_state.B[i * d + i] = 1.0;
    }

    m_state.generation = 0;
    m_state.eigen_eval_period = static_cast<uint32_t>(d / 10 + 1);
}

void CMAESOptimizer::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_best_fitness = -1e9;
    m_best_vector.assign(m_state.d, 0.0);
    m_total_evaluations = 0;
    m_state.sigma = m_config.initial_sigma;
    init_state();
}

std::vector<double> CMAESOptimizer::sample_candidate() {
    size_t d = m_state.d;
    std::vector<double> z(d);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (size_t i = 0; i < d; ++i) {
        z[i] = norm_dist(m_rng);
    }

    // Transform z via B and D: y = B * (D * z)
    std::vector<double> y(d, 0.0);
    for (size_t i = 0; i < d; ++i) {
        double scaled_z = m_state.D[i] * z[i];
        for (size_t j = 0; j < d; ++j) {
            y[j] += m_state.B[j * d + i] * scaled_z;
        }
    }

    // Candidate x = m + sigma * y
    std::vector<double> x(d);
    for (size_t i = 0; i < d; ++i) {
        x[i] = std::clamp(m_state.m[i] + m_state.sigma * y[i], -1.0, 1.0);
    }

    return x;
}

std::vector<std::vector<double>> CMAESOptimizer::ask() {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<std::vector<double>> population;
    population.reserve(m_state.lambda);

    for (size_t i = 0; i < m_state.lambda; ++i) {
        population.push_back(sample_candidate());
    }

    return population;
}

Status CMAESOptimizer::tell(
    std::span<const std::vector<double>> candidates,
    std::span<const double> fitness_scores
) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (candidates.size() != m_state.lambda || fitness_scores.size() != m_state.lambda) {
        return Status::ErrInvalidSearchBounds;
    }

    size_t d = m_state.d;
    size_t lambda = m_state.lambda;
    size_t mu = m_state.mu;

    m_total_evaluations += static_cast<uint32_t>(lambda);

    // 1. Sort candidate indices by fitness in descending order (higher score = better)
    std::vector<size_t> idx(lambda);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return fitness_scores[a] > fitness_scores[b];
    });

    // Update global best
    if (fitness_scores[idx[0]] > m_best_fitness) {
        m_best_fitness = fitness_scores[idx[0]];
        m_best_vector = candidates[idx[0]];
    }

    // 2. Compute new mean m_new
    std::vector<double> m_old = m_state.m;
    std::vector<double> m_new(d, 0.0);

    for (size_t i = 0; i < mu; ++i) {
        size_t cand_i = idx[i];
        for (size_t j = 0; j < d; ++j) {
            m_new[j] += m_state.weights[i] * candidates[cand_i][j];
        }
    }

    // Displacement vector: (m_new - m_old) / sigma
    std::vector<double> y_w(d);
    for (size_t j = 0; j < d; ++j) {
        y_w[j] = (m_new[j] - m_old[j]) / m_state.sigma;
    }

    // 3. Update evolution paths ps and pc
    double cs = m_state.cs;
    double cc = m_state.cc;
    double mueff = m_state.mueff;

    for (size_t j = 0; j < d; ++j) {
        m_state.ps[j] = (1.0 - cs) * m_state.ps[j] + std::sqrt(cs * (2.0 - cs) * mueff) * y_w[j];
        m_state.pc[j] = (1.0 - cc) * m_state.pc[j] + std::sqrt(cc * (2.0 - cc) * mueff) * y_w[j];
    }

    // 4. Adapt step size sigma
    double ps_norm = 0.0;
    for (size_t j = 0; j < d; ++j) {
        ps_norm += m_state.ps[j] * m_state.ps[j];
    }
    ps_norm = std::sqrt(ps_norm);

    double chiN = std::sqrt(static_cast<double>(d)) * (1.0 - 1.0 / (4.0 * static_cast<double>(d)) + 1.0 / (21.0 * static_cast<double>(d * d)));
    m_state.sigma *= std::exp((cs / m_state.damps) * (ps_norm / chiN - 1.0));

    // 5. Update Covariance Matrix C (Rank-1 and Rank-mu updates)
    double c1 = m_state.c1;
    double cmu = m_state.cmu;

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double rank1 = c1 * m_state.pc[i] * m_state.pc[j];
            double rank_mu = 0.0;

            for (size_t k = 0; k < mu; ++k) {
                size_t cand_k = idx[k];
                double y_k_i = (candidates[cand_k][i] - m_old[i]) / m_state.sigma;
                double y_k_j = (candidates[cand_k][j] - m_old[j]) / m_state.sigma;
                rank_mu += m_state.weights[k] * y_k_i * y_k_j;
            }

            m_state.C[i * d + j] = (1.0 - c1 - cmu) * m_state.C[i * d + j] + rank1 + cmu * rank_mu;
        }
    }

    m_state.m = m_new;
    m_state.generation++;

    // Periodically update matrix decomposition
    if (m_state.generation % m_state.eigen_eval_period == 0) {
        update_eigendecomposition();
    }

    return Status::Success;
}

void CMAESOptimizer::update_eigendecomposition() {
    size_t d = m_state.d;

    // Simplified diagonal approximation update for C = B * D^2 * B^T
    for (size_t i = 0; i < d; ++i) {
        m_state.D[i] = std::sqrt(std::max(1e-10, m_state.C[i * d + i]));
        m_state.B[i * d + i] = 1.0;
    }
}

std::vector<double> CMAESOptimizer::get_best_vector() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_best_vector;
}

} // namespace blackbox::optimizers