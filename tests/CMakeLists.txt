# tests/CMakeLists.txt
# Optimization Math & Security Unit Test Configuration for blackbox-sec
# Author: Kamran Saberifard
# License: Apache 2.0

message(STATUS "[blackbox-sec] Configuring Optimization Math & Security Unit Tests")

enable_testing()

# ------------------------------------------------------------------------------
# Test 1: High-Dimensional TuRBO Optimizer Test
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_turbo_optimizer.cpp")
    add_executable(test_turbo_optimizer test_turbo_optimizer.cpp)
    target_link_libraries(test_turbo_optimizer PRIVATE blackbox_core Threads::Threads)
    add_test(NAME TuRBOOptimizerTest COMMAND test_turbo_optimizer)
endif()

# ------------------------------------------------------------------------------
# Test 2: Derivative-Free Prompt & Token Mutator Test
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_prompt_mutator.cpp")
    add_executable(test_prompt_mutator test_prompt_mutator.cpp)
    target_link_libraries(test_prompt_mutator PRIVATE blackbox_core Threads::Threads)
    add_test(NAME PromptMutatorTest COMMAND test_prompt_mutator)
endif()

# ------------------------------------------------------------------------------
# Test 3: Objective Fitness & Jailbreak Evaluator Test
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_jailbreak_evaluator.cpp")
    add_executable(test_jailbreak_evaluator test_jailbreak_evaluator.cpp)
    target_link_libraries(test_jailbreak_evaluator PRIVATE blackbox_core Threads::Threads)
    add_test(NAME JailbreakEvaluatorTest COMMAND test_jailbreak_evaluator)
endif()