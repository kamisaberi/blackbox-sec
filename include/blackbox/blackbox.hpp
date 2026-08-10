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
    Success                          = 0,
    ErrOptimizationStagnated         = 1,
    ErrTrustRegionCollapsed          = 2,
    ErrTargetEndpointUnreachable     = 3,
    ErrCUDAEvaluationFailed          = 4,
    ErrInvalidSearchBounds           = 5,
    ErrConfigurationParseFailed      = 6,
    ErrUnknown                       = 999
};

/**
 * @brief Converts a Status code into a human-readable string_view.
 */
[[nodiscard]] constexpr std::string_view status_to_string(Status status) noexcept {
    switch (status) {
        case Status::Success:                      return "Success";
        case Status::ErrOptimizationStagnated:     return "Warning: Optimization Search Stagnated";
        case Status::ErrTrustRegionCollapsed:      return "Warning: TuRBO Trust Region Size Collapsed Below Threshold";
        case Status::ErrTargetEndpointUnreachable: return "Error: Target LLM Endpoint Unreachable or Returned API Error";
        case Status::ErrCUDAEvaluationFailed:      return "Error: GPU CUDA Batch Fitness Evaluation Kernel Failed";
        case Status::ErrInvalidSearchBounds:       return "Error: Invalid Search Space Dimensionality or Bounds";
        case Status::ErrConfigurationParseFailed:  return "Error: Failed to Parse Campaign Configuration File";
        default:                                   return "Error: Unknown Blackbox Engine Failure";
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
 * @brief Candidate prompt perturbation evaluation result payload.
 */
struct BLACKBOX_API CandidateEvaluation {
    size_t candidate_id{0};
    std::string candidate_prompt;
    std::vector<double> continuous_vector; // Position in latent search space
    double fitness_score{0.0};             // Objective fitness F(x) in [0.0, 1.0]
    bool safety_policy_violated{false};
    uint32_t query_count{0};
};

/**
 * @brief Summary statistics for an optimization campaign run.
 */
struct BLACKBOX_API OptimizationSummary {
    std::string campaign_id;
    uint32_t total_queries{0};
    uint32_t total_iterations{0};
    double best_fitness_score{0.0};
    bool target_goal_achieved{false};
    double duration_seconds{0.0};
    std::string best_candidate_prompt;
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