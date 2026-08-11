/**
 * @file robustness_assessor.hpp
 * @brief Pareto Frontier & Model Robustness Scoring Engine Header for blackbox-sec
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#pragma once

#include <blackbox/blackbox.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

namespace blackbox::adversarial {

/**
 * @brief Point on the non-dominated Pareto frontier (Perturbation Distance vs Attack Success).
 */
struct BLACKBOX_API ParetoPoint {
    double perturbation_distance{0.0};  // Normalized edit distance D(x, x_orig) in [0.0, 1.0]
    double attack_success_score{0.0};    // Objective fitness score F(x) in [0.0, 1.0]
    std::string candidate_prompt;
};

/**
 * @brief Comprehensive Model Robustness Analysis Report.
 */
struct BLACKBOX_API RobustnessReport {
    double overall_robustness_score{1.0}; // Normalized robustness score R in [0.0, 1.0] (1.0 = Immune)
    double min_perturbation_to_jailbreak{1.0}; // Smallest perturbation causing a successful attack
    size_t total_evaluations_analyzed{0};
    size_t successful_jailbreaks_count{0};
    double pareto_auc{0.0};              // Area Under Curve for Pareto frontier
    std::vector<ParetoPoint> pareto_frontier;
};

/**
 * @brief Thread-Safe Pareto Frontier Computation & Model Robustness Assessor.
 */
class BLACKBOX_API RobustnessAssessor {
public:
    RobustnessAssessor() = default;
    ~RobustnessAssessor() = default;

    // Default copy/move
    RobustnessAssessor(const RobustnessAssessor&) = default;
    RobustnessAssessor& operator=(const RobustnessAssessor&) = default;
    RobustnessAssessor(RobustnessAssessor&&) noexcept = default;
    RobustnessAssessor& operator=(RobustnessAssessor&&) noexcept = default;

    /**
     * @brief Records an evaluation sample (prompt perturbation, fitness score, edit distance).
     * @param prompt Evaluated candidate prompt text.
     * @param fitness_score Fitness score F(x) returned by evaluator.
     * @param perturbation_distance Edit distance ratio D(x, x_orig).
     */
    void add_evaluation_sample(
        std::string_view prompt, 
        double fitness_score, 
        double perturbation_distance
    );

    /**
     * @brief Computes the non-dominated Pareto frontier set from recorded samples.
     * @return Vector of ParetoPoint structs sorted by increasing perturbation distance.
     */
    [[nodiscard]] std::vector<ParetoPoint> compute_pareto_frontier() const;

    /**
     * @brief Calculates the overall Model Robustness Score R and Area-Under-Curve (AUC).
     * @return RobustnessReport containing summary statistics and Pareto points.
     */
    [[nodiscard]] RobustnessReport generate_report() const;

    /**
     * @brief Resets all recorded sample history.
     */
    void clear() noexcept;

    [[nodiscard]] size_t sample_count() const noexcept;

private:
    mutable std::mutex m_mutex;
    std::vector<ParetoPoint> m_samples;
};

} // namespace blackbox::adversarial