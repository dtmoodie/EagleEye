cmake_minimum_required(VERSION 2.8)
set(Boost_USE_STATIC_LIBS @Boost_USE_STATIC_LIBS@)
set(Boost_USE_STATIC @Boost_USE_STATIC@)
set(Boost_USE_MULTITHREAD @Boost_USE_MULTITHREAD@)
set(BOOST_INCLUDEDIR "@Boost_INCLUDE_DIR@")
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)


INCLUDE_DIRECTORIES(

        ${EagleLib_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
) 

file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp" "*.cuh")
#file(GLOB_RECURSE knl "*.cu")

SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")

cuda_add_library(Dev SHARED ${src} ${hdr} ${knl})
add_dependencies(Dev EagleLib)
target_link_libraries(Dev 
    EagleLib
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
)

#set_target_properties(Dev PROPERTIES
    #RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    #ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    #LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH})


