/**
 * @file test_turbo_optimizer.cpp
 * @brief Unit Tests for TuRBO Optimization Engine in blackbox-sec
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#include <blackbox/blackbox.hpp>
#include <blackbox/optimizers/turbo_optimizer.hpp>

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

namespace {

void test_turbo_initialization_and_sampling() {
    std::cout << "[TEST] Running TuRBO Initialization & Candidate Sampling Test...\n";

    blackbox::optimizers::TuRBOConfig cfg{};
    cfg.dimension = 16;
    cfg.num_trust_regions = 2;
    cfg.batch_size = 8;

    blackbox::optimizers::TuRBOOptimizer optimizer(cfg);

    assert(optimizer.config().dimension == 16);
    assert(optimizer.get_best_fitness() < -1e8);

    // Ask for candidates
    auto candidates = optimizer.ask();

    assert(candidates.size() == 8);
    assert(candidates[0].size() == 16);

    // Verify candidate values fall within [-1.0, 1.0] bounds
    for (const auto& vec : candidates) {
        for (double val : vec) {
            assert(val >= -1.0 && val <= 1.0);
        }
    }

    std::cout << "\033[1;32m[PASS] TuRBO Initialization & Sampling Verified!\033[0m\n";
}

void test_turbo_tell_and_expansion() {
    std::cout << "[TEST] Running TuRBO Tell & State Update Test...\n";

    blackbox::optimizers::TuRBOConfig cfg{};
    cfg.dimension = 8;
    cfg.num_trust_regions = 1;
    cfg.batch_size = 4;

    blackbox::optimizers::TuRBOOptimizer optimizer(cfg);

    auto candidates = optimizer.ask();
    std::vector<double> scores = {0.10, 0.45, 0.82, 0.30}; // Max score = 0.82

    blackbox::Status status = optimizer.tell(candidates, scores);
    assert(status == blackbox::Status::Success);

    assert(std::abs(optimizer.get_best_fitness() - 0.82) < 1e-5);
    assert(optimizer.total_evaluations() == 4);

    auto best_vec = optimizer.get_best_vector();
    assert(best_vec.size() == 8);

    std::cout << "\033[1;32m[PASS] TuRBO Tell & Global Best Tracking Verified!\033[0m\n";
}

void test_turbo_reset() {
    std::cout << "[TEST] Running TuRBO Reset Test...\n";

    blackbox::optimizers::TuRBOConfig cfg{};
    cfg.dimension = 8;

    blackbox::optimizers::TuRBOOptimizer optimizer(cfg);
    auto candidates = optimizer.ask();
    std::vector<double> scores(candidates.size(), 0.90);

    optimizer.tell(candidates, scores);
    assert(optimizer.get_best_fitness() > 0.80);

    optimizer.reset_all_regions();
    assert(optimizer.get_best_fitness() < -1e8);
    assert(optimizer.total_evaluations() == 0);

    std::cout << "\033[1;32m[PASS] TuRBO Reset Verified!\033[0m\n";
}

} // anonymous namespace

int main() {
    std::cout << "\033[1;36m===================================================\033[0m\n";
    std::cout << "\033[1;36m blackbox-sec TuRBO Optimizer Unit Tests           \033[0m\n";
    std::cout << "\033[1;36m===================================================\033[0m\n\n";

    test_turbo_initialization_and_sampling();
    test_turbo_tell_and_expansion();
    test_turbo_reset();

    std::cout << "\n\033[1;32mAll TuRBO Optimizer Unit Tests PASSED!\033[0m\n";
    return 0;
}