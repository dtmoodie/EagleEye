CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(Parameters)

FIND_PACKAGE(Boost)
FIND_PACKAGE(Qt5 COMPONENTS Gui Widgets)
FIND_PACKAGE(OpenCV COMPONENTS core)

FILE(GLOB_RECURSE hdr "include/*.hpp" "include/*.h")
FILE(GLOB_RECURSE src "src/*.c" "src/*.cpp")

IF(Qt5_FOUND)
ADD_DEFINITIONS(-DQt5_FOUND)
ENDIF()
IF(Boost_FOUND)
ADD_DEFINITIONS(-DBoost_FOUND)
ENDIF()


INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/include" ${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
SET(Parameters_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

add_executable(Parameter_test "src/parameter_test.cpp")
#add_library(Parameters ${hdr} ${src})

 