cmake_minimum_required(VERSION 2.8)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)


INCLUDE_DIRECTORIES(
        ${EagleLib_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)
file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp" "*.cuh")
file(GLOB_RECURSE knl "*.cu")

#SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")

cuda_add_library(Segmentation SHARED ${src} ${hdr} ${knl})
add_dependencies(Segmentation EagleLib)
target_link_libraries(Segmentation
    EagleLib
    ${OpenCV_LIBS}
)

set_target_properties(Segmentation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
)
