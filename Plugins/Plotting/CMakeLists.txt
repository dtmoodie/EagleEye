cmake_minimum_required(VERSION 2.8)
PROJECT(Plotting)
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "include/*.cuh")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB forms "*UIs/*.ui")
CMAKE_POLICY(SET CMP0020 NEW)
find_package(Qt5 COMPONENTS Core Widgets PrintSupport OpenGL)
IF(Qt5_FOUND)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudabgsegm)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
set(BOOST_REQUIRED_MODULES system chrono thread date_time)
 find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
ADD_DEFINITIONS( -DHAVE_QT5 )
QT5_WRAP_UI(UIS ${forms})
QT5_WRAP_CPP(MOC ${hdr})
LINK_DIRECTORIES(${LINK_DIRS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${QT_INCLUDES}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5PrintSupport_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
)

#add_definitions(-DQT_INCLUDE_DIR=${Qt5Core_INCLUDE_DIRS})
add_definitions(-DPROJECT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")



add_library(Plotting SHARED ${src} ${hdr} ${knl} ${UIS} ${MOC})
add_dependencies(Plotting EagleLib)

RCC_LINK_LIB(Plotting 
	EagleLib 
	Qt5::Core 
	Qt5::Widgets 
	Qt5::OpenGL
	Qt5::PrintSupport
	)
INCLUDE(../PluginTemplate.cmake)
ENDIF(Qt5_FOUND)
