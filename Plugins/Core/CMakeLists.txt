
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)

ADD_DEFINITIONS(${DEFS})
LINK_DIRECTORIES(${LINK_DIRS})
INCLUDE_DIRECTORIES(
    ${EagleLib_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
) 

file(GLOB_RECURSE src "*.cpp" ".cu")
file(GLOB_RECURSE hdr "*.h" "*.hpp")

IF(WIN32)

ELSE()
    #SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")
ENDIF()

cuda_add_library(Core SHARED ${src} ${hdr})
MESSAGE(STATUS
"=========== Core ============
    Defs: ${DEFS}
")
add_dependencies(Core EagleLib)
target_link_libraries(Core 
    EagleLib
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
)

