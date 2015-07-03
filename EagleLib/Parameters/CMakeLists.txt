CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(Parameters)

FIND_PACKAGE(Boost)
FIND_PACKAGE(Qt5 COMPONENTS Gui Widgets)
FIND_PACKAGE(OpenCV COMPONENTS core)

FILE(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
FILE(GLOB_RECURSE src "src/*.c" "src/*.cpp")

IF(Qt5_FOUND)
ADD_DEFINITIONS(-DQt5_FOUND)
ENDIF()
IF(Boost_FOUND)
ADD_DEFINITIONS(-DBoost_FOUND)
ENDIF()


INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/src" ${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
SET(Parameters_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
add_library(libParameter ${hdr} ${src})
add_executable(Parameter_test "test/parameter_test.cpp")
TARGET_LINK_LIBRARIES(Parameter_test libParameter ${OpenCV_LIBS})
ADD_DEPENDENCIES(Parameter_test libParameter)


  