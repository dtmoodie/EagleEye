PROJECT(Core)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)
  include_directories(${CUDA_TOOLKIT_INCLUDE})


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

ADD_DEFINITIONS(-DPROJECT_INCLUDES="${CMAKE_CURRENT_SOURCE_DIR}")

SET(knl "")


file(GLOB_RECURSE knl "*.cu")
file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp")

IF(WIN32)

ELSE()
    #SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
ENDIF()

cuda_add_library(Core SHARED ${src} ${hdr} ${knl})
ocv_add_precompiled_header_to_target(Core nodes/precompiled.hpp)
add_dependencies(Core EagleLib)
RCC_LINK_LIB(Core EagleLib ${OpenCV_LIBS} ${CUDA_CUBLAS_LIBRARIES} MetaObject)

INCLUDE(../PluginTemplate.cmake)

