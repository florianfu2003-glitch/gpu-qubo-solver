cmake_minimum_required(VERSION 3.26)

project(QUBOBruteForcing LANGUAGES CXX CUDA)

# -------------------------------------------------------------------
# C++ & CUDA Standards
# -------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Find CUDA (Lichtenberg GPU nodes, CUDA 12.5)
# -------------------------------------------------------------------
find_package(CUDAToolkit QUIET)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA FOUND → GPU version will be built")
    set(WITH_CUDA TRUE)
else()
    message(STATUS "CUDA NOT FOUND → Building CPU-only version")
    set(WITH_CUDA FALSE)
endif()

# -------------------------------------------------------------------
# Executable
# -------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    main.cpp
    gpu_brute_force.cu        # CUDA code
    gpu_brute_force.h
    cpu_brute_force.h
    cuda_debug.h
    cuda_timer.h
    cuda_util.h
    datatypes.h
    matrix.h
    matrix_reader.h
    qubo_brute_forcer.h 
    qubo_energy.h
    state_vector.h
    timer.h
)

# -------------------------------------------------------------------
# CUDA Settings (only if CUDA exists)
# -------------------------------------------------------------------
if (WITH_CUDA)
    enable_language(CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA)

    # Important for Lichtenberg (Tesla T4)
    set(CMAKE_CUDA_ARCHITECTURES 75)

    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Link CUDA Runtime
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

# -------------------------------------------------------------------
# Fix GCC 8 filesystem (must link manually!)
# -------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
