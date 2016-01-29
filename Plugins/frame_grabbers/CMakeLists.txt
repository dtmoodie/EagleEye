PROJECT(frame_grabbers)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgcodecs cudacodec)
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

IF(WIN32)
    file(GLOB_RECURSE knl "*.cu")
ENDIF(WIN32)
file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp")

IF(WIN32)

ELSE()
    #SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")
ENDIF()

cuda_add_library(frame_grabbers SHARED ${src} ${hdr} ${knl})

add_dependencies(frame_grabbers EagleLib)
RCC_LINK_LIB(frame_grabbers EagleLib ${OpenCV_LIBS} ${CUDA_CUBLAS_LIBRARIES})

INCLUDE(../PluginTemplate.cmake)

