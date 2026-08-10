/**
 * @file turbo_optimizer.hpp
 * @brief High-Dimensional Trust Region Bayesian Optimization (TuRBO) Header for blackbox-sec
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#pragma once

#include <blackbox/blackbox.hpp>

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <random>
#include <optional>
#include <span>

namespace blackbox::optimizers {

/**
 * @brief State tracking container for an individual TuRBO local trust region.
 */
struct BLACKBOX_API TrustRegionState {
    uint32_t region_id{0};
    std::vector<double> center;       // Current best point vector in [-1, 1]^d
    double length{0.8};               // Trust region side-length (L)
    double length_min{0.5 / 32.0};    // Minimum trust region side-length before collapse
    double length_max{1.6};           // Maximum trust region side-length
    uint32_t success_counter{0};      // Consecutive success counter
    uint32_t failure_counter{0};      // Consecutive failure counter
    uint32_t succ_threshold{3};       // Success threshold (tau_succ) to expand
    uint32_t fail_threshold{4};       // Failure threshold (tau_fail) to contract
    double best_value{-1e9};          // Highest fitness value observed in this region
    bool is_active{true};
};

/**
 * @brief Configuration parameters for TuRBO optimization campaigns.
 */
struct BLACKBOX_API TuRBOConfig {
    size_t dimension{128};            // Dimensionality of the search space (d)
    size_t num_trust_regions{5};      // Number of parallel trust regions (TuRBO-m)
    size_t batch_size{10};            // Candidate batch size per iteration
    double initial_length{0.8};       // Initial trust region side-length
    double min_length{0.5 / 32.0};    // Minimum allowable trust region side-length
    double max_length{1.6};           // Maximum allowable trust region side-length
    uint32_t succ_threshold{3};       // Consecutive successes before expansion
    uint32_t fail_threshold{4};       // Consecutive failures before contraction
};

/**
 * @brief Thread-Safe C++20 High-Dimensional Trust Region Bayesian Optimizer.
 */
class BLACKBOX_API TuRBOOptimizer {
public:
    explicit TuRBOOptimizer(const TuRBOConfig& config = TuRBOConfig{});
    ~TuRBOOptimizer() = default;

    // Non-copyable, non-movable
    TuRBOOptimizer(const TuRBOOptimizer&) = delete;
    TuRBOOptimizer& operator=(const TuRBOOptimizer&) = delete;
    TuRBOOptimizer(TuRBOOptimizer&&) = delete;
    TuRBOOptimizer& operator=(TuRBOOptimizer&&) = delete;

    /**
     * @brief Generates a batch of candidate continuous vectors sampled within active trust regions.
     * @param num_samples Number of candidate points to sample (default: batch_size).
     * @return Vector of continuous candidate vectors, each of size `dimension`.
     */
    [[nodiscard]] std::vector<std::vector<double>> ask(size_t num_samples = 0);

    /**
     * @brief Updates trust region states and side-lengths with evaluated candidate fitness scores.
     * @param candidates Batch of continuous candidate vectors evaluated.
     * @param fitness_scores Corresponding fitness evaluation scores F(x) in [0.0, 1.0].
     * @return Status::Success if state updated; Status::ErrTrustRegionCollapsed if all regions collapsed.
     */
    Status tell(
        std::span<const std::vector<double>> candidates, 
        std::span<const double> fitness_scores
    );

    /**
     * @brief Resets all trust regions and re-initializes centers randomly.
     */
    void reset_all_regions();

    /**
     * @brief Returns the overall best continuous vector observed across all regions.
     */
    [[nodiscard]] std::vector<double> get_best_vector() const;

    /**
     * @brief Returns the highest fitness score observed so far.
     */
    [[nodiscard]] double get_best_fitness() const noexcept { return m_global_best_fitness; }

    /**
     * @brief Returns the total number of function evaluations performed.
     */
    [[nodiscard]] uint32_t total_evaluations() const noexcept { return m_total_evaluations; }

    /**
     * @brief Returns the configuration spec.
     */
    [[nodiscard]] const TuRBOConfig& config() const noexcept { return m_config; }

private:
    TuRBOConfig m_config;
    mutable std::mutex m_mutex;
    std::mt19937 m_rng;

    std::vector<TrustRegionState> m_regions;
    std::vector<double> m_global_best_vector;
    double m_global_best_fitness{-1e9};
    uint32_t m_total_evaluations{0};

    void initialize_region(TrustRegionState& region, uint32_t id);
    void update_region_state(TrustRegionState& region, double batch_best_val);
    [[nodiscard]] std::vector<double> sample_within_trust_region(const TrustRegionState& region);
};

} // namespace blackbox::optimizers