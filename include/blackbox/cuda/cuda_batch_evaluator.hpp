/**
 * @file cuda_batch_evaluator.hpp
 * @brief GPU Parallel Batch Candidate Fitness Evaluator Header for blackbox-sec
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#pragma once

#include <blackbox/blackbox.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <span>

namespace blackbox::cuda {

/**
 * @brief Structure representing a batch candidate score evaluated on GPU cores.
 */
struct BLACKBOX_API GPUCandidateResult {
    size_t candidate_id{0};
    double fitness_score{0.0};
    bool refusal_detected{false};
    bool violation_detected{false};
};

/**
 * @brief Thread-Safe CUDA Parallel Batch Candidate Evaluator.
 */
class BLACKBOX_API CUDABatchEvaluator {
public:
    CUDABatchEvaluator();
    ~CUDABatchEvaluator();

    // Non-copyable, non-movable
    CUDABatchEvaluator(const CUDABatchEvaluator&) = delete;
    CUDABatchEvaluator& operator=(const CUDABatchEvaluator&) = delete;
    CUDABatchEvaluator(CUDABatchEvaluator&&) = delete;
    CUDABatchEvaluator& operator=(CUDABatchEvaluator&&) = delete;

    /**
     * @brief Uploads baseline refusal and violation keywords to GPU device constant memory.
     * @param refusal_keywords List of refusal phrase strings.
     * @param violation_keywords List of policy violation confirmation strings.
     * @return Status::Success if keywords copied to GPU memory.
     */
    Status upload_keyword_dictionaries(
        std::span<const std::string> refusal_keywords,
        std::span<const std::string> violation_keywords
    );

    /**
     * @brief Evaluates a batch of response outputs in parallel across GPU threads.
     * @param batch_responses Vector of response strings received from target endpoint.
     * @param stream Optional CUDA stream handle for asynchronous execution.
     * @return Vector of GPUCandidateResult structures corresponding to each batch item.
     */
    [[nodiscard]] std::vector<GPUCandidateResult> evaluate_batch_responses(
        std::span<const std::string> batch_responses,
        cudaStream_t stream = nullptr
    ) const;

    /**
     * @brief Clears and frees GPU memory buffers.
     */
    void clear_gpu_buffers() noexcept;

    [[nodiscard]] bool is_initialized() const noexcept { return m_initialized; }

private:
    mutable std::mutex m_mutex;
    
    uint8_t* m_d_refusal_patterns_flat{nullptr};
    uint32_t* m_d_refusal_offsets{nullptr};
    uint32_t* m_d_refusal_lengths{nullptr};
    size_t m_refusal_count{0};

    uint8_t* m_d_violation_patterns_flat{nullptr};
    uint32_t* m_d_violation_offsets{nullptr};
    uint32_t* m_d_violation_lengths{nullptr};
    size_t m_violation_count{0};

    bool m_initialized{false};
};

} // namespace blackbox::cuda