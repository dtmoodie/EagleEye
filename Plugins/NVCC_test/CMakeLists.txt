cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-std=c++11")
message(STATUS
"============================ nvcc_test ==================================
  Cuda includes: ${CUDA_INCLUDE_DIRS}

==========================================================================")
include_directories(
 ${EagleLib_INCLUDE_DIRECTORIES}
 ${CUDA_INCLUDE_DIRS}
)
cuda_add_library(nvcc_test SHARED nvcc_test.cu nvcc_test.h nvcc_test.cpp nvcc_test.cuh)

set_target_properties(nvcc_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
)
