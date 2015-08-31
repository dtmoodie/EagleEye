cmake_minimum_required(VERSION 2.8)


CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
FIND_PACKAGE(PCL REQUIRED COMPONENTS common visualization)
LINK_DIRECTORIES(${LINK_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

INCLUDE_DIRECTORIES(
        ${EagleLib_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

SET(PCL_BIN_DIR_DEBUG "${PCL_DIR}/../bin" CACHE PATH "Path to PCL debug binaries")
SET(PCL_BIN_DIR_RELEASE "${PCL_DIR}/../bin" CACHE PATH "Path to PCL release binaries")
FIND_PACKAGE(VTK REQUIRED)
SET(VTK_BIN_DIR "${VTK_LIBRARY_DIRS}/../bin" CACHE PATH "Path to VTK binaries")

FILE(GLOB_RECURSE SRC "*.cpp")
FILE(GLOB_RECURSE HDR "*.h")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_library(PCL_bridge SHARED ${SRC} ${HDR})
add_dependencies(PCL_bridge EagleLib)
target_link_libraries(PCL_bridge 
	EagleLib 
	${EagleLib_LIBRARIES}
	${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES}
	)

