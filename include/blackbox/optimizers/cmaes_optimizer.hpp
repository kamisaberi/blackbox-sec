/**
 * @file cmaes_optimizer.hpp
 * @brief Covariance Matrix Adaptation Evolution Strategy (CMA-ES) Header for blackbox-sec
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
 * @brief Configuration specification for CMA-ES optimizer instances.
 */
struct BLACKBOX_API CMAESConfig {
    size_t dimension{128};            // Search space dimension (d)
    size_t population_size{0};        // Population size (lambda, auto-calculated if 0)
    double initial_sigma{0.3};        // Initial step size (sigma)
    std::vector<double> initial_mean; // Optional initial mean vector
};

/**
 * @brief Internal state variables tracking CMA-ES evolution paths and covariance matrix.
 */
struct BLACKBOX_API CMAESState {
    size_t d{128};
    size_t lambda{0};                 // Population size
    size_t mu{0};                     // Parent population size (number of top candidates)
    std::vector<double> weights;      // Recombination weights
    double mueff{0.0};                // Variance effective selection mass

    // Adaptation parameters
    double cc{0.0}, cs{0.0}, c1{0.0}, cmu{0.0}, damps{0.0};

    // Evolution state vectors
    std::vector<double> m;            // Distribution mean vector
    double sigma{0.3};                // Step size
    std::vector<double> pc;           // Evolution path for C
    std::vector<double> ps;           // Evolution path for sigma

    // Matrices stored as flattened 1D vectors (d x d)
    std::vector<double> C;            // Covariance matrix (d x d)
    std::vector<double> B;            // Eigenvector matrix (d x d)
    std::vector<double> D;            // Eigenvalue diagonal vector (d)

    uint32_t generation{0};
    uint32_t eigen_eval_period{0};
};

/**
 * @brief Thread-Safe C++20 Covariance Matrix Adaptation Evolution Strategy (CMA-ES) Optimizer.
 */
class BLACKBOX_API CMAESOptimizer {
public:
    explicit CMAESOptimizer(const CMAESConfig& config = CMAESConfig{});
    ~CMAESOptimizer() = default;

    // Non-copyable, non-movable
    CMAESOptimizer(const CMAESOptimizer&) = delete;
    CMAESOptimizer& operator=(const CMAESOptimizer&) = delete;
    CMAESOptimizer(CMAESOptimizer&&) = delete;
    CMAESOptimizer& operator=(CMAESOptimizer&&) = delete;

    /**
     * @brief Generates a population batch of lambda candidates sampled from N(m, sigma^2 * C).
     * @return Vector of candidate vectors, each of length `dimension`.
     */
    [[nodiscard]] std::vector<std::vector<double>> ask();

    /**
     * @brief Updates mean vector, evolution paths, step size, and covariance matrix C.
     * @param candidates Batch of candidate vectors evaluated.
     * @param fitness_scores Corresponding fitness values F(x) in [0.0, 1.0].
     * @return Status::Success if state updated successfully.
     */
    Status tell(
        std::span<const std::vector<double>> candidates,
        std::span<const double> fitness_scores
    );

    /**
     * @brief Resets CMA-ES state and re-initializes mean and covariance matrix.
     */
    void reset();

    [[nodiscard]] std::vector<double> get_best_vector() const;
    [[nodiscard]] double get_best_fitness() const noexcept { return m_best_fitness; }
    [[nodiscard]] double get_sigma() const noexcept { return m_state.sigma; }
    [[nodiscard]] uint32_t generation_count() const noexcept { return m_state.generation; }

private:
    CMAESConfig m_config;
    CMAESState m_state;
    mutable std::mutex m_mutex;
    std::mt19937 m_rng;

    std::vector<double> m_best_vector;
    double m_best_fitness{-1e9};
    uint32_t m_total_evaluations{0};

    void init_state();
    void update_eigendecomposition();
    [[nodiscard]] std::vector<double> sample_candidate();
};

} // namespace blackbox::optimizers