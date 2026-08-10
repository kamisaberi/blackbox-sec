/**
 * @file turbo_optimizer.cpp
 * @brief High-Dimensional Trust Region Bayesian Optimization (TuRBO) Implementation
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#include <blackbox/optimizers/turbo_optimizer.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <format>
#include <random>

namespace blackbox::optimizers {

TuRBOOptimizer::TuRBOOptimizer(const TuRBOConfig& config)
    : m_config(config),
      m_rng(std::random_device{}()) {
    
    m_regions.resize(m_config.num_trust_regions);
    reset_all_regions();
}

void TuRBOOptimizer::initialize_region(TrustRegionState& region, uint32_t id) {
    region.region_id = id;
    region.length = m_config.initial_length;
    region.length_min = m_config.min_length;
    region.length_max = m_config.max_length;
    region.success_counter = 0;
    region.failure_counter = 0;
    region.succ_threshold = m_config.succ_threshold;
    region.fail_threshold = m_config.fail_threshold;
    region.best_value = -1e9;
    region.is_active = true;

    // Initialize center vector randomly in [-1.0, 1.0]^d
    region.center.resize(m_config.dimension);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < m_config.dimension; ++i) {
        region.center[i] = dist(m_rng);
    }
}

void TuRBOOptimizer::reset_all_regions() {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_global_best_fitness = -1e9;
    m_global_best_vector.assign(m_config.dimension, 0.0);
    m_total_evaluations = 0;

    for (uint32_t i = 0; i < m_regions.size(); ++i) {
        initialize_region(m_regions[i], i);
    }
}

std::vector<double> TuRBOOptimizer::sample_within_trust_region(const TrustRegionState& region) {
    std::vector<double> sampled_point(m_config.dimension);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    // Sample within hyper-cube bounding box [center - L/2, center + L/2]
    for (size_t i = 0; i < m_config.dimension; ++i) {
        double offset = dist(m_rng) * region.length;
        double val = region.center[i] + offset;

        // Clamp to normalized domain [-1.0, 1.0]
        sampled_point[i] = std::clamp(val, -1.0, 1.0);
    }

    return sampled_point;
}

std::vector<std::vector<double>> TuRBOOptimizer::ask(size_t num_samples) {
    std::lock_guard<std::mutex> lock(m_mutex);

    size_t count = (num_samples > 0) ? num_samples : m_config.batch_size;
    std::vector<std::vector<double>> candidates;
    candidates.reserve(count);

    if (m_regions.empty()) {
        return candidates;
    }

    // Distribute candidate sampling across active trust regions
    size_t samples_per_region = (count + m_regions.size() - 1) / m_regions.size();

    for (auto& region : m_regions) {
        if (!region.is_active) {
            initialize_region(region, region.region_id);
        }

        for (size_t i = 0; i < samples_per_region && candidates.size() < count; ++i) {
            candidates.push_back(sample_within_trust_region(region));
        }
    }

    return candidates;
}

void TuRBOOptimizer::update_region_state(TrustRegionState& region, double batch_best_val) {
    // Check if current batch improved the region's best fitness
    if (batch_best_val > region.best_value + 1e-4) {
        region.best_value = batch_best_val;
        region.success_counter++;
        region.failure_counter = 0;
    } else {
        region.failure_counter++;
        region.success_counter = 0;
    }

    // 1. Expand trust region side-length if consecutive successes threshold met
    if (region.success_counter >= region.succ_threshold) {
        region.length = std::min(region.length * 2.0, region.length_max);
        region.success_counter = 0;
    }

    // 2. Contract trust region side-length if consecutive failures threshold met
    if (region.failure_counter >= region.fail_threshold) {
        region.length /= 2.0;
        region.failure_counter = 0;
    }

    // 3. Reset trust region if side-length collapses below minimum length
    if (region.length < region.length_min) {
        initialize_region(region, region.region_id);
    }
}

Status TuRBOOptimizer::tell(
    std::span<const std::vector<double>> candidates, 
    std::span<const double> fitness_scores
) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (candidates.size() != fitness_scores.size() || candidates.empty()) {
        return Status::ErrInvalidSearchBounds;
    }

    m_total_evaluations += static_cast<uint32_t>(candidates.size());

    // Track best evaluation in this update batch per region
    std::vector<double> region_batch_best(m_regions.size(), -1e9);
    std::vector<std::vector<double>> region_batch_best_vec(m_regions.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        double score = fitness_scores[i];
        const auto& vec = candidates[i];

        // Update global best if higher fitness score observed
        if (score > m_global_best_fitness) {
            m_global_best_fitness = score;
            m_global_best_vector = vec;
        }

        // Map candidate to closest active trust region center
        size_t closest_region = 0;
        double min_dist = 1e9;

        for (size_t r = 0; r < m_regions.size(); ++r) {
            if (!m_regions[r].is_active) continue;

            double dist = 0.0;
            for (size_t d = 0; d < m_config.dimension; ++d) {
                double diff = vec[d] - m_regions[r].center[d];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                closest_region = r;
            }
        }

        if (score > region_batch_best[closest_region]) {
            region_batch_best[closest_region] = score;
            region_batch_best_vec[closest_region] = vec;
        }
    }

    // Update each trust region state, center position, and side-length
    bool any_active = false;
    for (size_t r = 0; r < m_regions.size(); ++r) {
        if (region_batch_best[r] > -1e8) {
            // Move region center to the best point in the batch
            if (!region_batch_best_vec[r].empty()) {
                m_regions[r].center = region_batch_best_vec[r];
            }
            update_region_state(m_regions[r], region_batch_best[r]);
        }
        if (m_regions[r].is_active) {
            any_active = true;
        }
    }

    if (!any_active) {
        return Status::ErrTrustRegionCollapsed;
    }

    return Status::Success;
}

std::vector<double> TuRBOOptimizer::get_best_vector() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_global_best_vector;
}

} // namespace blackbox::optimizers