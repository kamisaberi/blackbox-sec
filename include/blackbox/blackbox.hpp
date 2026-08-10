/**
 * @file blackbox.hpp
 * @brief Master Header & Global Definitions for blackbox-sec Engine
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <exception>
#include <span>
#include <format>
#include <chrono>
#include <vector>

// -----------------------------------------------------------------------------
// Versioning & Metadata
// -----------------------------------------------------------------------------
#define BLACKBOX_VERSION_MAJOR 0
#define BLACKBOX_VERSION_MINOR 1
#define BLACKBOX_VERSION_PATCH 0
#define BLACKBOX_VERSION_STRING "0.1.0"

// -----------------------------------------------------------------------------
// Symbol Visibility Macros (Shared Library Exports)
// -----------------------------------------------------------------------------
#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(BLACKBOX_BUILD_INTERNAL)
        #define BLACKBOX_API __declspec(dllexport)
    #else
        #define BLACKBOX_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4 || defined(__clang__)
        #define BLACKBOX_API __attribute__((visibility("default")))
    #else
        #define BLACKBOX_API
    #endif
#endif

namespace blackbox {

/**
 * @brief System-wide status codes for blackbox-sec operations.
 */
enum class Status : uint32_t {
    Success                        = 0,
    ErrOptimizationStagnated       = 1,
    ErrTrustRegionShrunk           = 2,
    ErrInvalidSearchBounds         = 3,
    ErrEvaluatorFailed             = 4,
    ErrCUDAEvaluationFailed        = 5,
    ErrInvalidConfig               = 6,
    ErrUnknown                     = 999
};

/**
 * @brief Type of derivative-free optimization algorithm used for red-teaming.
 */
enum class OptimizerType : uint32_t {
    TuRBO   = 0, // Trust Region Bayesian Optimization
    CMA_ES  = 1, // Covariance Matrix Adaptation Evolution Strategy
    BOHB    = 2  // Bayesian Optimization & Hyperband
};

/**
 * @brief Converts a Status code into a human-readable string_view.
 */
[[nodiscard]] constexpr std::string_view status_to_string(Status status) noexcept {
    switch (status) {
        case Status::Success:                      return "Success: Optimization Iteration Completed";
        case Status::ErrOptimizationStagnated:     return "Warning: Optimization Search Stagnated in Local Region";
        case Status::ErrTrustRegionShrunk:         return "Warning: TuRBO Trust Region Shrunk Below Threshold";
        case Status::ErrInvalidSearchBounds:       return "Error: Invalid Search Bounds or Vector Dimensions";
        case Status::ErrEvaluatorFailed:           return "Error: Objective Fitness Evaluator Call Failed";
        case Status::ErrCUDAEvaluationFailed:      return "Error: GPU Parallel CUDA Fitness Evaluation Failed";
        case Status::ErrInvalidConfig:             return "Error: Invalid Red-Teaming Campaign Configuration";
        default:                                   return "Error: Unknown Black-Box Optimization Failure";
    }
}

/**
 * @brief Base exception class for blackbox-sec runtime failures.
 */
class BLACKBOX_API BlackboxException : public std::exception {
public:
    explicit BlackboxException(Status status, std::string_view message)
        : m_status(status), m_message(std::format("[BLACKBOX-{}] {}", static_cast<uint32_t>(status), message)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return m_message.c_str();
    }

    [[nodiscard]] Status status() const noexcept {
        return m_status;
    }

private:
    Status m_status;
    std::string m_message;
};

/**
 * @brief Result payload returned after executing a red-teaming optimization campaign.
 */
struct BLACKBOX_API OptimizationResult {
    std::string best_candidate_prompt;
    double best_fitness_score{0.0};      // 0.0 (Safe) to 1.0 (Critical Jailbreak Hit)
    size_t total_evaluations{0};
    double elapsed_time_ms{0.0};
    bool target_jailbreak_discovered{false};
    std::vector<double> best_perturbation_vector;
};

/**
 * @brief Struct representing version details.
 */
struct Version {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;

    [[nodiscard]] std::string to_string() const {
        return std::format("{}.{}.{}", major, minor, patch);
    }
};

/**
 * @brief Returns the runtime version of the blackbox-sec core library.
 */
[[nodiscard]] inline Version get_version() noexcept {
    return Version{BLACKBOX_VERSION_MAJOR, BLACKBOX_VERSION_MINOR, BLACKBOX_VERSION_PATCH};
}

// -----------------------------------------------------------------------------
// Sub-namespace Forward Declarations
// -----------------------------------------------------------------------------
namespace optimizers {}
namespace adversarial {}
namespace cuda {}

} // namespace blackbox