cmake_minimum_required(VERSION 2.8)
PROJECT(QCustomPlot)

FILE(GLOB_RECURSE qcustomplot "*")

CMAKE_POLICY(SET CMP0020 NEW)
find_package(Qt5 COMPONENTS Core Widgets PrintSupport OpenGL)
    
IF(Qt5_FOUND)
set(CMAKE_AUTOMOC ON)
ADD_DEFINITIONS( -DHAVE_QT5 )
#QT5_WRAP_CPP(MOC ${qcustomplot})


INCLUDE_DIRECTORIES(
    ${QT_INCLUDES}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5PrintSupport_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
)

add_definitions(-DQCUSTOMPLOT_COMPILE_LIBRARY)
add_definitions(-DPROJECT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


add_library(QCustomPlot SHARED ${qcustomplot} ${MOC})
TARGET_LINK_LIBRARIES(QCustomPlot 
    Qt5::Core 
	Qt5::Widgets 
	Qt5::OpenGL
	Qt5::PrintSupport)

ENDIF(Qt5_FOUND)
