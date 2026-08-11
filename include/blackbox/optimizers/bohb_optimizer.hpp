/**
 * @file bohb_optimizer.hpp
 * @brief Bayesian Optimization and Hyperband (BOHB) Early-Stopping Engine Header
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#pragma once

#include <blackbox/blackbox.hpp>

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <optional>
#include <span>

namespace blackbox::optimizers {

/**
 * @brief Representation of a candidate evaluation at a specific budget rung.
 */
struct BLACKBOX_API BOHBCandidate {
    uint64_t candidate_id{0};
    std::vector<double> vector;
    double fitness_score{0.0};
    uint32_t allocated_budget{1};
    uint32_t current_rung{0};
    bool is_promoted{false};
};

/**
 * @brief Configuration specification for BOHB successive halving brackets.
 */
struct BLACKBOX_API BOHBConfig {
    size_t dimension{128};            // Search space dimension
    uint32_t min_budget{1};           // Minimum resource budget per candidate (e.g. 1 iteration)
    uint32_t max_budget{81};          // Maximum resource budget per candidate (e.g. 81 iterations)
    uint32_t eta{3};                  // Halving factor (eta = 3 means top 1/3 promoted)
    size_t num_brackets{4};           // Number of Hyperband brackets
};

/**
 * @brief Representation of an individual Successive Halving Bracket in BOHB.
 */
struct BLACKBOX_API BOHBBracket {
    uint32_t bracket_id{0};
    uint32_t s_max{0};
    uint32_t current_rung{0};
    uint32_t current_budget{1};
    size_t current_n_candidates{0};
    std::vector<BOHBCandidate> active_candidates;
};

/**
 * @brief Thread-Safe C++20 Bayesian Optimization and Hyperband (BOHB) Engine.
 */
class BLACKBOX_API BOHBOptimizer {
public:
    explicit BOHBOptimizer(const BOHBConfig& config = BOHBConfig{});
    ~BOHBOptimizer() = default;

    // Non-copyable, non-movable
    BOHBOptimizer(const BOHBOptimizer&) = delete;
    BOHBOptimizer& operator=(const BOHBOptimizer&) = delete;
    BOHBOptimizer(BOHBOptimizer&&) = delete;
    BOHBOptimizer& operator=(BOHBOptimizer&&) = delete;

    /**
     * @brief Generates a candidate evaluation batch assigned to the current successive halving budget.
     * @return Vector of BOHBCandidate objects specifying vector coordinates and target budgets.
     */
    [[nodiscard]] std::vector<BOHBCandidate> ask();

    /**
     * @brief Updates BOHB bracket state with evaluated fitness scores and executes successive halving promotions.
     * @param evaluated_candidates Vector of candidates with updated fitness scores.
     * @return Status::Success if bracket state was successfully updated.
     */
    Status tell(std::span<const BOHBCandidate> evaluated_candidates);

    /**
     * @brief Calculates the top 1/eta candidate fraction eligible for promotion to the next budget rung.
     * @param candidates Current candidate list at rung.
     * @param num_to_promote Number of candidates to promote.
     * @return Vector of promoted candidate IDs.
     */
    [[nodiscard]] static std::vector<uint64_t> calculate_promotions(
        std::span<const BOHBCandidate> candidates, 
        size_t num_to_promote
    );

    /**
     * @brief Resets all Hyperband brackets and clears candidate histories.
     */
    void reset();

    [[nodiscard]] double get_best_fitness() const noexcept { return m_best_fitness; }
    [[nodiscard]] std::vector<double> get_best_vector() const;
    [[nodiscard]] uint32_t total_budget_spent() const noexcept { return m_total_budget_spent; }
    [[nodiscard]] const BOHBConfig& config() const noexcept { return m_config; }

private:
    BOHBConfig m_config;
    mutable std::mutex m_mutex;

    std::vector<BOHBBracket> m_brackets;
    size_t m_current_bracket_idx{0};
    uint64_t m_next_candidate_id{1};

    double m_best_fitness{-1e9};
    std::vector<double> m_best_vector;
    uint32_t m_total_budget_spent{0};

    void initialize_brackets();
    void advance_bracket_state(BOHBBracket& bracket);
};

} // namespace blackbox::optimizers