PROJECT(SLAM)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)

set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}	
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/dtam/Cpp
)
#ADD_SUBDIRECTORY("dtam/cpp")
ADD_SUBDIRECTORY("lsd_slam/lsd_slam_core")

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp" "src/*.cuh")

IF(WIN32)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
ELSE()
    #SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")
ENDIF()

cuda_add_library(SLAM SHARED ${src} ${hdr} ${knl})

add_dependencies(SLAM EagleLib)
add_dependencies(SLAM OpenDTAM)
RCC_LINK_LIB(SLAM 
	${OpenCV_LIBS}
	${CUDA_CUBLAS_LIBRARIES} 
	EagleLib
	OpenDTAM
)

INCLUDE(../PluginTemplate.cmake)

