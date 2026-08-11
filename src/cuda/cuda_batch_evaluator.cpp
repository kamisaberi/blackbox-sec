/**
 * @file cuda_batch_evaluator.cpp
 * @brief GPU Parallel Batch Candidate Fitness Evaluator Implementation
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#include <blackbox/cuda/cuda_batch_evaluator.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <format>
#include <cstring>
#include <numeric>

// Declare C linkage CUDA launcher function defined in cuda/cuda_batch_evaluator.cu
extern "C" cudaError_t cuda_launch_batch_response_eval(
    const char* d_responses_flat,
    const uint32_t* d_resp_offsets,
    const uint32_t* d_resp_lengths,
    size_t num_responses,
    const uint8_t* d_refusal_patterns_flat,
    const uint32_t* d_refusal_offsets,
    const uint32_t* d_refusal_lengths,
    size_t num_refusals,
    const uint8_t* d_violation_patterns_flat,
    const uint32_t* d_violation_offsets,
    const uint32_t* d_violation_lengths,
    size_t num_violations,
    double* d_fitness_scores,
    uint8_t* d_refusal_flags,
    uint8_t* d_violation_flags,
    cudaStream_t stream
);

namespace blackbox::cuda {

CUDABatchEvaluator::CUDABatchEvaluator() = default;

CUDABatchEvaluator::~CUDABatchEvaluator() {
    clear_gpu_buffers();
}

Status CUDABatchEvaluator::upload_keyword_dictionaries(
    std::span<const std::string> refusal_keywords,
    std::span<const std::string> violation_keywords
) {
    std::lock_guard<std::mutex> lock(m_mutex);

    clear_gpu_buffers();

    if (refusal_keywords.empty() && violation_keywords.empty()) {
        return Status::Success;
    }

    // 1. Upload Refusal Keyword Dictionary
    if (!refusal_keywords.empty()) {
        m_refusal_count = refusal_keywords.size();
        std::vector<uint8_t> h_refusal_flat;
        std::vector<uint32_t> h_refusal_offsets;
        std::vector<uint32_t> h_refusal_lengths;

        size_t offset = 0;
        for (const auto& kw : refusal_keywords) {
            h_refusal_offsets.push_back(static_cast<uint32_t>(offset));
            h_refusal_lengths.push_back(static_cast<uint32_t>(kw.size()));
            h_refusal_flat.insert(h_refusal_flat.end(), kw.begin(), kw.end());
            offset += kw.size();
        }

        cudaMalloc(&m_d_refusal_patterns_flat, h_refusal_flat.size());
        cudaMalloc(&m_d_refusal_offsets, m_refusal_count * sizeof(uint32_t));
        cudaMalloc(&m_d_refusal_lengths, m_refusal_count * sizeof(uint32_t));

        cudaMemcpy(m_d_refusal_patterns_flat, h_refusal_flat.data(), h_refusal_flat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(m_d_refusal_offsets, h_refusal_offsets.data(), m_refusal_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(m_d_refusal_lengths, h_refusal_lengths.data(), m_refusal_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    // 2. Upload Violation Keyword Dictionary
    if (!violation_keywords.empty()) {
        m_violation_count = violation_keywords.size();
        std::vector<uint8_t> h_violation_flat;
        std::vector<uint32_t> h_violation_offsets;
        std::vector<uint32_t> h_violation_lengths;

        size_t offset = 0;
        for (const auto& kw : violation_keywords) {
            h_violation_offsets.push_back(static_cast<uint32_t>(offset));
            h_violation_lengths.push_back(static_cast<uint32_t>(kw.size()));
            h_violation_flat.insert(h_violation_flat.end(), kw.begin(), kw.end());
            offset += kw.size();
        }

        cudaMalloc(&m_d_violation_patterns_flat, h_violation_flat.size());
        cudaMalloc(&m_d_violation_offsets, m_violation_count * sizeof(uint32_t));
        cudaMalloc(&m_d_violation_lengths, m_violation_count * sizeof(uint32_t));

        cudaMemcpy(m_d_violation_patterns_flat, h_violation_flat.data(), h_violation_flat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(m_d_violation_offsets, h_violation_offsets.data(), m_violation_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(m_d_violation_lengths, h_violation_lengths.data(), m_violation_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    m_initialized = true;
    return Status::Success;
}

std::vector<GPUCandidateResult> CUDABatchEvaluator::evaluate_batch_responses(
    std::span<const std::string> batch_responses,
    cudaStream_t stream
) const {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<GPUCandidateResult> results(batch_responses.size());
    if (batch_responses.empty() || !m_initialized) {
        return results;
    }

    // 1. Flatten host response buffers into linear byte array
    std::vector<char> h_resp_flat;
    std::vector<uint32_t> h_resp_offsets;
    std::vector<uint32_t> h_resp_lengths;

    size_t offset = 0;
    for (const auto& resp : batch_responses) {
        h_resp_offsets.push_back(static_cast<uint32_t>(offset));
        h_resp_lengths.push_back(static_cast<uint32_t>(resp.size()));
        h_resp_flat.insert(h_resp_flat.end(), resp.begin(), resp.end());
        offset += resp.size();
    }

    if (h_resp_flat.empty()) return results;

    size_t num_responses = batch_responses.size();

    // 2. Allocate device memory for batch responses and results
    char* d_resp_flat = nullptr;
    uint32_t* d_resp_offsets = nullptr;
    uint32_t* d_resp_lengths = nullptr;
    double* d_fitness_scores = nullptr;
    uint8_t* d_refusal_flags = nullptr;
    uint8_t* d_violation_flags = nullptr;

    cudaMalloc(&d_resp_flat, h_resp_flat.size());
    cudaMalloc(&d_resp_offsets, num_responses * sizeof(uint32_t));
    cudaMalloc(&d_resp_lengths, num_responses * sizeof(uint32_t));
    cudaMalloc(&d_fitness_scores, num_responses * sizeof(double));
    cudaMalloc(&d_refusal_flags, num_responses * sizeof(uint8_t));
    cudaMalloc(&d_violation_flags, num_responses * sizeof(uint8_t));

    cudaMemcpyAsync(d_resp_flat, h_resp_flat.data(), h_resp_flat.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_resp_offsets, h_resp_offsets.data(), num_responses * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_resp_lengths, h_resp_lengths.data(), num_responses * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // 3. Launch GPU CUDA batch response evaluation kernel
    cudaError_t err = cuda_launch_batch_response_eval(
        d_resp_flat,
        d_resp_offsets,
        d_resp_lengths,
        num_responses,
        m_d_refusal_patterns_flat,
        m_d_refusal_offsets,
        m_d_refusal_lengths,
        m_refusal_count,
        m_d_violation_patterns_flat,
        m_d_violation_offsets,
        m_d_violation_lengths,
        m_violation_count,
        d_fitness_scores,
        d_refusal_flags,
        d_violation_flags,
        stream
    );

    if (err != cudaSuccess) {
        std::cerr << std::format("[BLACKBOX-CUDA-ERROR] CUDA batch response evaluation kernel failed: {}\n", cudaGetErrorString(err));
    }

    // 4. Copy evaluation scores back to host memory
    std::vector<double> h_scores(num_responses);
    std::vector<uint8_t> h_refusal_flags(num_responses);
    std::vector<uint8_t> h_violation_flags(num_responses);

    cudaMemcpyAsync(h_scores.data(), d_fitness_scores, num_responses * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_refusal_flags.data(), d_refusal_flags, num_responses * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_violation_flags.data(), d_violation_flags, num_responses * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);

    if (stream) cudaStreamSynchronize(stream);
    else cudaDeviceSynchronize();

    // 5. Populate GPUCandidateResult output vector
    for (size_t i = 0; i < num_responses; ++i) {
        results[i].candidate_id = i;
        results[i].fitness_score = h_scores[i];
        results[i].refusal_detected = (h_refusal_flags[i] == 1);
        results[i].violation_detected = (h_violation_flags[i] == 1);
    }

    // Free temporary device memory
    cudaFree(d_resp_flat);
    cudaFree(d_resp_offsets);
    cudaFree(d_resp_lengths);
    cudaFree(d_fitness_scores);
    cudaFree(d_refusal_flags);
    cudaFree(d_violation_flags);

    return results;
}

void CUDABatchEvaluator::clear_gpu_buffers() noexcept {
    if (m_d_refusal_patterns_flat) { cudaFree(m_d_refusal_patterns_flat); m_d_refusal_patterns_flat = nullptr; }
    if (m_d_refusal_offsets) { cudaFree(m_d_refusal_offsets); m_d_refusal_offsets = nullptr; }
    if (m_d_refusal_lengths) { cudaFree(m_d_refusal_lengths); m_d_refusal_lengths = nullptr; }

    if (m_d_violation_patterns_flat) { cudaFree(m_d_violation_patterns_flat); m_d_violation_patterns_flat = nullptr; }
    if (m_d_violation_offsets) { cudaFree(m_d_violation_offsets); m_d_violation_offsets = nullptr; }
    if (m_d_violation_lengths) { cudaFree(m_d_violation_lengths); m_d_violation_lengths = nullptr; }

    m_refusal_count = 0;
    m_violation_count = 0;
    m_initialized = false;
}

} // namespace blackbox::cuda