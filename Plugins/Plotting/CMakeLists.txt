cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "include/*.cuh")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB forms "*UIs/*.ui")

find_package(Qt5 REQUIRED Core Widgets PrintSupport OpenGL)


find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudabgsegm)
set(BOOST_REQUIRED_MODULES system chrono thread date_time)
 find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
ADD_DEFINITIONS( -DHAVE_QT5 )
QT5_WRAP_UI(UIS ${forms})
QT5_WRAP_CPP(MOC ${src} ${hdr})


INCLUDE_DIRECTORIES(
    ${EagleLib_INCLUDE_DIRECTORIES}
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
message(STATUS
"
==============================================
            Plotting
    Qt core include dir: ${Qt5Core_INCLUDE_DIRS}


	 

")

add_library(Plotting SHARED ${src} ${hdr} ${knl} ${UIS} ${MOC})
add_dependencies(Plotting EagleLib)
set_target_properties(Plotting PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
            ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
            LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
)
target_link_libraries(Plotting 
	EagleLib 
	Qt5::Core 
	Qt5::Widgets 
	Qt5::OpenGL
	Qt5::PrintSupport
	)

#add_custom_command(TARGET Plotting POST_BUILD COMMAND ${CMAKE_COMMAND} -E COPY_FILE)
#target_link_libraries(Plotting EagleLib)
