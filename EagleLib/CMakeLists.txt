cmake_minimum_required(VERSION 2.8)
project(EagleLib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if(WIN32)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif(WIN32)


set(OPENCV_STATIC ON)
find_package(OpenCV REQUIRED core imgproc highgui cuda cudaimgproc cudawarping cudafeatures2d)
message(${OpenCV_CONFIG_PATH})

find_package(CUDA REQUIRED)

set(Boost_USE_STATIC_LIBS @Boost_USE_STATIC_LIBS@)
set(Boost_USE_STATIC @Boost_USE_STATIC@)
set(Boost_USE_MULTITHREAD @Boost_USE_MULTITHREAD@)
set(BOOST_INCLUDEDIR "@Boost_INCLUDE_DIR@")
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


file(GLOB_RECURSE src "src/*.cpp" "src/nodes/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/nodes/*.h")
add_definitions(-DWIN32)
add_definitions(-D_WIN32)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("include")

set(EagleLib_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
if(WIN32)
	if(CMAKE_BUILD_TYPE EQUAL "Debug")
		set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_DEBUG_POSTFIX}.lib PARENT_SCOPE)
		message(${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_DEBUG_POSTFIX}.dll)
	else()
		set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_RELEASE_POSTFIX}.lib PARENT_SCOPE)
		message(${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_RELEASE_POSTFIX}.dll)
	endif(CMAKE_BUILD_TYPE EQUAL "Debug")
else()
	if(CMAKE_BUILD_TYPE EQUAL "Debug")
		set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/libEagleLib${CMAKE_DEBUG_POSTFIX}.so PARENT_SCOPE)
		else()
		set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/libEagleLib${CMAKE_RELEASE_POSTFIX}.so PARENT_SCOPE)
		endif(CMAKE_BUILD_TYPE EQUAL "Debug")
endif(WIN32)


add_library(EagleLib SHARED 
	${src} 
	${hdr}
)


TARGET_LINK_LIBRARIES(EagleLib 
	${Boost_LIBRARIES} 
	${OpenCV_LIBS} 
)
