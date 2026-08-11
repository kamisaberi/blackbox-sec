/**
 * @file bohb_optimizer.cpp
 * @brief Bayesian Optimization and Hyperband (BOHB) Early-Stopping Engine Implementation
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#include <blackbox/optimizers/bohb_optimizer.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <format>

namespace blackbox::optimizers {

BOHBOptimizer::BOHBOptimizer(const BOHBConfig& config)
    : m_config(config) {
    
    m_best_vector.assign(m_config.dimension, 0.0);
    initialize_brackets();
}

void BOHBOptimizer::initialize_brackets() {
    m_brackets.clear();

    // Calculate maximum number of rungs / brackets
    // s_max = floor(log_eta(max_budget / min_budget))
    double eta_dbl = static_cast<double>(m_config.eta);
    double ratio = static_cast<double>(m_config.max_budget) / static_cast<double>(m_config.min_budget);
    uint32_t s_max = static_cast<uint32_t>(std::floor(std::log(ratio) / std::log(eta_dbl)));

    m_brackets.resize(s_max + 1);

    for (uint32_t s = 0; s <= s_max; ++s) {
        BOHBBracket bracket{};
        bracket.bracket_id = s;
        bracket.s_max = s_max;
        bracket.current_rung = 0;

        // n = ceil((s_max + 1) / (s + 1) * eta^s)
        double n_dbl = std::ceil((static_cast<double>(s_max + 1) / static_cast<double>(s + 1)) * std::pow(eta_dbl, s));
        bracket.current_n_candidates = static_cast<size_t>(n_dbl);

        // initial_budget = max_budget * eta^(-s)
        double r_dbl = static_cast<double>(m_config.max_budget) * std::pow(eta_dbl, -static_cast<double>(s));
        bracket.current_budget = std::max(m_config.min_budget, static_cast<uint32_t>(r_dbl));

        m_brackets[s] = std::move(bracket);
    }

    m_current_bracket_idx = 0;
}

void BOHBOptimizer::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_best_fitness = -1e9;
    m_best_vector.assign(m_config.dimension, 0.0);
    m_total_budget_spent = 0;
    m_next_candidate_id = 1;

    initialize_brackets();
}

std::vector<BOHBCandidate> BOHBOptimizer::ask() {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<BOHBCandidate> sample_batch;

    if (m_brackets.empty()) {
        return sample_batch;
    }

    auto& current_bracket = m_brackets[m_current_bracket_idx];

    // If current rung has no active candidates, initialize candidate pool for rung 0
    if (current_bracket.active_candidates.empty() && current_bracket.current_rung == 0) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        current_bracket.active_candidates.reserve(current_bracket.current_n_candidates);

        for (size_t i = 0; i < current_bracket.current_n_candidates; ++i) {
            BOHBCandidate cand{};
            cand.candidate_id = m_next_candidate_id++;
            cand.allocated_budget = current_bracket.current_budget;
            cand.current_rung = 0;
            cand.is_promoted = false;

            // Generate normalized initial search vector
            cand.vector.resize(m_config.dimension);
            for (size_t d = 0; d < m_config.dimension; ++d) {
                cand.vector[d] = dist(rng);
            }

            current_bracket.active_candidates.push_back(std::move(cand));
        }
    }

    return current_bracket.active_candidates;
}

std::vector<uint64_t> BOHBOptimizer::calculate_promotions(
    std::span<const BOHBCandidate> candidates, 
    size_t num_to_promote
) {
    if (candidates.empty() || num_to_promote == 0) {
        return {};
    }

    std::vector<size_t> indices(candidates.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort candidates by fitness score in descending order
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return candidates[a].fitness_score > candidates[b].fitness_score;
    });

    std::vector<uint64_t> promoted_ids;
    promoted_ids.reserve(std::min(num_to_promote, candidates.size()));

    for (size_t i = 0; i < num_to_promote && i < candidates.size(); ++i) {
        promoted_ids.push_back(candidates[indices[i]].candidate_id);
    }

    return promoted_ids;
}

void BOHBOptimizer::advance_bracket_state(BOHBBracket& bracket) {
    uint32_t eta = m_config.eta;
    size_t current_count = bracket.active_candidates.size();
    size_t num_to_promote = current_count / eta;

    if (num_to_promote == 0 || bracket.current_budget >= m_config.max_budget) {
        // Bracket completed -> advance to next bracket or loop around
        bracket.active_candidates.clear();
        bracket.current_rung = 0;
        m_current_bracket_idx = (m_current_bracket_idx + 1) % m_brackets.size();
        return;
    }

    // Identify top 1/eta candidates
    auto promoted_ids = calculate_promotions(bracket.active_candidates, num_to_promote);
    std::unordered_map<uint64_t, bool> promo_map;
    for (uint64_t id : promoted_ids) {
        promo_map[id] = true;
    }

    // Filter candidate list to keep only promoted candidates
    std::vector<BOHBCandidate> next_rung_candidates;
    next_rung_candidates.reserve(promoted_ids.size());

    // Advance budget allocation (budget_next = budget * eta)
    uint32_t next_budget = std::min(m_config.max_budget, bracket.current_budget * eta);

    for (auto& cand : bracket.active_candidates) {
        if (promo_map.contains(cand.candidate_id)) {
            cand.current_rung++;
            cand.allocated_budget = next_budget;
            cand.is_promoted = true;
            next_rung_candidates.push_back(std::move(cand));
        }
    }

    bracket.active_candidates = std::move(next_rung_candidates);
    bracket.current_rung++;
    bracket.current_budget = next_budget;
    bracket.current_n_candidates = bracket.active_candidates.size();
}

Status BOHBOptimizer::tell(std::span<const BOHBCandidate> evaluated_candidates) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_brackets.empty() || evaluated_candidates.empty()) {
        return Status::ErrInvalidSearchBounds;
    }

    auto& current_bracket = m_brackets[m_current_bracket_idx];

    // Map evaluation scores back to active candidate records
    for (const auto& eval : evaluated_candidates) {
        m_total_budget_spent += eval.allocated_budget;

        // Update global best tracking
        if (eval.fitness_score > m_best_fitness) {
            m_best_fitness = eval.fitness_score;
            m_best_vector = eval.vector;
        }

        for (auto& cand : current_bracket.active_candidates) {
            if (cand.candidate_id == eval.candidate_id) {
                cand.fitness_score = eval.fitness_score;
                break;
            }
        }
    }

    // Advance successive halving rung state inside the bracket
    advance_bracket_state(current_bracket);

    return Status::Success;
}

std::vector<double> BOHBOptimizer::get_best_vector() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_best_vector;
}

} // namespace blackbox::optimizers