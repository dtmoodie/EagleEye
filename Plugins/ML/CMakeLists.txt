PROJECT(ML)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo videoio)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

ADD_DEFINITIONS(${DEFS})
LINK_DIRECTORIES(${LINK_DIRS})
INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE knl "*.cu")
file(GLOB_RECURSE knl_hdr "*.cuh")
file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp")
IF(UNIX)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "-std=c++11;--expt-relaxed-constexpr;${CUDA_NVCC_FLAGS}")
ENDIF()
cuda_add_library(ML SHARED ${src} ${hdr} ${knl} ${knl_hdr})

RCC_LINK_LIB(ML aquila_core aquila_utilities ${OpenCV_LIBS} ${CUDA_CUBLAS_LIBRARIES} RuntimeObjectSystem)

INCLUDE(../PluginTemplate.cmake)

