PROJECT(nvcc_test)
find_package(CUDA REQUIRED)
cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0012 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()


set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono regex)
set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_VERBOSE_BUILD ON)
find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES}) 
LINK_DIRECTORIES(${LINK_DIRS})
IF(WIN32)

ELSE()
set(CUDA_NVCC_FLAGS "-std=c++11")
ENDIF(WIN32)

message(STATUS
"=============== nvcc_test ============
  Cuda includes: ${CUDA_INCLUDE_DIRS}

  Cuda libs: ${CUDA_LIBRARIES}

")
include_directories(
 ${EagleLib_INCLUDE_DIRECTORIES}
 ${CUDA_INCLUDE_DIRS}
)
ADD_DEFINITIONS(-DNVCC_TEST_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}")

IF(WIN32)
	LINK_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
ELSE(WIN32)
	IF(EAGLE_LIB_JETSON)
		LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib)
	ELSE()
		LINK_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
	ENDIF(EAGLE_LIB_JETSON)
ENDIF(WIN32)

cuda_add_library(nvcc_test SHARED nvcc_test.cu nvcc_test.h nvcc_test.cpp nvcc_test.cuh)
target_link_libraries(nvcc_test EagleLib ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
INCLUDE(../PluginTemplate.cmake)
