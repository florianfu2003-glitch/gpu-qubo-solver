cmake_minimum_required(VERSION 3.26)

project(QUBOBruteForcing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA FOUND → GPU version will be built")
    set(WITH_CUDA TRUE)
else()
    message(STATUS "CUDA NOT FOUND → Building CPU-only version")
    set(WITH_CUDA FALSE)
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    gpu_brute_force.cu        
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

if (WITH_CUDA)
    enable_language(CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA)

    set(CMAKE_CUDA_ARCHITECTURES 75)

    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
