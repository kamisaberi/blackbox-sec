# CMakeLists.txt
# Root C++20 & CUDA Build System for blackbox-sec
# Author: Kamran Saberifard
# License: Apache 2.0

cmake_minimum_required(VERSION 3.20)

project(blackbox-sec
    VERSION 0.1.0
    DESCRIPTION "Parallel Adversarial Red-Teaming & Model Robustness Engine"
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------------------------
# 1. C++ Standard & Compiler Optimization Flags
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build type for maximum optimization throughput
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Target modern CUDA architectures (Ampere, Ada, Hopper, Blackwell ready)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# Apply compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -O3
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wnull-dereference
        -Wconversion
        -fstrict-aliasing
    )
    if(NOT CMAKE_CROSSCOMPILING)
        add_compile_options(-march=native)
    endif()
endif()

# ------------------------------------------------------------------------------
# 2. Find Dependencies (OpenSSL 3.0, CUDAToolkit, Threads, PkgConfig)
# ------------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

find_package(OpenSSL 3.0 REQUIRED)
message(STATUS "[blackbox-sec] Found OpenSSL: Version ${OPENSSL_VERSION}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "[blackbox-sec] Found CUDA Toolkit: Version ${CUDAToolkit_VERSION}")

# ------------------------------------------------------------------------------
# 3. Core C++/CUDA Library Target (`blackbox_core`)
# ------------------------------------------------------------------------------
file(GLOB_RECURSE BLACKBOX_CXX_SOURCES
    "src/*.cpp"
)

file(GLOB_RECURSE BLACKBOX_CUDA_SOURCES
    "cuda/*.cu"
)

add_library(blackbox_core SHARED
    ${BLACKBOX_CXX_SOURCES}
    ${BLACKBOX_CUDA_SOURCES}
)

target_include_directories(blackbox_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OPENSSL_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(blackbox_core
    PUBLIC
        OpenSSL::Crypto
        CUDA::cudart
        CUDA::cuda_driver
        Threads::Threads
)

target_compile_definitions(blackbox_core
    PRIVATE
        BLACKBOX_BUILD_INTERNAL=1
)

# ------------------------------------------------------------------------------
# 4. CLI Executable Target (`blackbox-redteam`)
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmd/blackbox-redteam/main.cpp")
    add_executable(blackbox-redteam cmd/blackbox-redteam/main.cpp)
    target_link_libraries(blackbox-redteam PRIVATE blackbox_core)
endif()

# ------------------------------------------------------------------------------
# 5. Unit & Optimization Testing
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()