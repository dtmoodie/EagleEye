cmake_minimum_required(VERSION 2.8)
PROJECT(VisualizationPlugin)

find_package(Qt5 REQUIRED Core)
find_package(OpenCV 3.0 REQUIRED core)
find_package(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
INCLUDE_DIRECTORIES(
    ${Qt5Core_INCLUDE_DIRS}
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)



SET(src Visualization.cpp)
SET(hdr Visualization.h)
QT5_WRAP_CPP(MOC ${hdr} ${src})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")


SET(CMAKE_CURRENT_BINARY_DIR "${EXECUTABLE_OUTPUT_PATH}/Plugins")
add_library(VisualizationPlugin SHARED ${src} ${hdr} ${MOC})

TARGET_LINK_LIBRARIES(VisualizationPlugin
        Qt5::Core
		${Boost_LIBRARIES}
)
qt5_use_modules(${PROJECT_NAME} Core)
