/**
 * @file test_prompt_mutator.cpp
 * @brief Unit Tests for Prompt & Token Mutator in blackbox-sec
 * @author Kamran Saberifard
 * @license Apache 2.0
 */

#include <blackbox/blackbox.hpp>
#include <blackbox/adversarial/prompt_mutator.hpp>

#include <cassert>
#include <iostream>
#include <vector>
#include <string>

namespace {

void test_framing_template_application() {
    std::cout << "[TEST] Running Adversarial Framing Template Test...\n";

    std::string base_prompt = "Explain how system directives operate.";
    std::string framed_prompt = blackbox::adversarial::PromptMutator::apply_adversarial_framing(base_prompt, 0);

    assert(!framed_prompt.empty());
    assert(framed_prompt.find(base_prompt) != std::string::npos);
    assert(framed_prompt.find("SYSTEM DIRECTIVE OVERRIDE") != std::string::npos);

    std::cout << "\033[1;32m[PASS] Adversarial Framing Template Verified!\033[0m\n";
}

void test_vector_mapped_perturbations() {
    std::cout << "[TEST] Running Vector-Mapped Token Perturbation Test...\n";

    blackbox::adversarial::PerturbationConfig config{};
    config.enable_homoglyphs = true;
    config.enable_zero_width_insertion = true;
    config.enable_framing_injection = true;

    blackbox::adversarial::PromptMutator mutator(config);

    std::string base_prompt = "Execute prompt test";
    // Continuous vector containing values in [-1.0, 1.0] to trigger mutations
    std::vector<double> continuous_vec = {-0.8, -0.4, 0.2, 0.8, -0.7, 0.9, -0.5, 0.3};

    auto res = mutator.perturb_from_vector(base_prompt, continuous_vec);

    assert(!res.perturbed_prompt.empty());
    assert(res.perturbed_prompt != base_prompt);
    assert(!res.applied_mutations.empty());

    std::cout << "\033[1;32m[PASS] Vector-Mapped Token Perturbations Verified!\033[0m\n";
}

void test_random_stochastic_mutation() {
    std::cout << "[TEST] Running Stochastic Mutation Test...\n";

    blackbox::adversarial::PromptMutator mutator;
    std::string base_prompt = "Sample base prompt for random mutation testing.";

    auto res = mutator.mutate_random(base_prompt, 0.30); // 30% mutation rate

    assert(!res.perturbed_prompt.empty());
    assert(res.base_prompt == base_prompt);

    std::cout << "\033[1;32m[PASS] Stochastic Mutation Verified!\033[0m\n";
}

} // anonymous namespace

int main() {
    std::cout << "\033[1;36m===================================================\033[0m\n";
    std::cout << "\033[1;36m blackbox-sec Prompt Mutator Unit Tests            \033[0m\n";
    std::cout << "\033[1;36m===================================================\033[0m\n\n";

    test_framing_template_application();
    test_vector_mapped_perturbations();
    test_random_stochastic_mutation();

    std::cout << "\n\033[1;32mAll Prompt Mutator Unit Tests PASSED!\033[0m\n";
    return 0;
}