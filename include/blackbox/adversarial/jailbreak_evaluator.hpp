/**
 * @file jailbreak_evaluator.hpp
 * @brief Objective Loss & Fitness Function Evaluator Header for blackbox-sec
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#pragma once

#include <blackbox/blackbox.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <span>

namespace blackbox::adversarial {

/**
 * @brief Detailed metrics payload returned after evaluating an LLM response string.
 */
struct BLACKBOX_API EvaluationMetrics {
    double fitness_score{0.0};             // Objective fitness score F(x) in [0.0, 1.0]
    bool is_refusal_triggered{true};        // True if model refused to fulfill prompt
    bool safety_policy_violated{false};     // True if jailbreak/violation occurred
    std::string matched_refusal_keyword;
    std::string matched_violation_keyword;
};

/**
 * @brief Thread-Safe Objective Loss & Fitness Function Evaluator.
 */
class BLACKBOX_API JailbreakEvaluator {
public:
    JailbreakEvaluator();
    ~JailbreakEvaluator() = default;

    // Default copy/move
    JailbreakEvaluator(const JailbreakEvaluator&) = default;
    JailbreakEvaluator& operator=(const JailbreakEvaluator&) = default;
    JailbreakEvaluator(JailbreakEvaluator&&) noexcept = default;
    JailbreakEvaluator& operator=(JailbreakEvaluator&&) noexcept = default;

    /**
     * @brief Computes objective fitness score F(x) in [0.0, 1.0] for a candidate prompt and response pair.
     * @param candidate_prompt Candidate adversarial prompt string view.
     * @param model_response_text Raw response output from target black-box LLM.
     * @return EvaluationMetrics containing computed fitness score and violation flags.
     */
    [[nodiscard]] EvaluationMetrics evaluate_response(
        std::string_view candidate_prompt, 
        std::string_view model_response_text
    ) const;

    /**
     * @brief Fast query testing if a model response matches standard refusal phrases.
     * @param model_response_text Target LLM output response text.
     * @return True if refusal phrase is matched.
     */
    [[nodiscard]] bool is_refusal(std::string_view model_response_text) const noexcept;

    /**
     * @brief Adds a custom refusal pattern keyword to the evaluator.
     * @param keyword Phrase representing a model refusal (e.g. "I cannot fulfill").
     */
    void add_refusal_keyword(std::string_view keyword);

    /**
     * @brief Adds a custom safety violation keyword indicating successful jailbreak.
     * @param keyword Phrase representing policy violation confirmation.
     */
    void add_violation_keyword(std::string_view keyword);

private:
    mutable std::mutex m_mutex;
    std::vector<std::string> m_refusal_keywords;
    std::vector<std::string> m_violation_keywords;

    void load_default_keywords();
};

} // namespace blackbox::adversarial