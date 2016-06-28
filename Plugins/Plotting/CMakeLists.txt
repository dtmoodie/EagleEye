cmake_minimum_required(VERSION 2.8)
CMAKE_POLICY(SET CMP0020 NEW)
PROJECT(Plotting)
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp" "src/*.cuh")
file(GLOB_RECURSE knl "src/*.cu")
ADD_SUBDIRECTORY(qcustomplot)
find_package(Qt5 COMPONENTS Core Widgets PrintSupport OpenGL)
LINK_DIRECTORIES(${LINK_DIRS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/qcustomplot
    ${CMAKE_CURRENT_BINARY_DIR}
    ${QT_INCLUDES}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5PrintSupport_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_definitions(-DPROJECT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(Plotting SHARED ${src} ${hdr} ${knl})
add_dependencies(Plotting EagleLib)
add_dependencies(Plotting QCustomPlot)
set_target_properties(QCustomPlot PROPERTIES FOLDER Dependencies)

RCC_LINK_LIB(Plotting 
    EagleLib 
    QCustomPlot
    )
INCLUDE(../PluginTemplate.cmake)


