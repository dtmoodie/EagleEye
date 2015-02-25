cmake_minimum_required(VERSION 2.8)
project(EagleLib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(OPENCV_STATIC OFF)
find_package(OpenCV 3.0 REQUIRED core imgproc highgui cuda cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
message(${OpenCV_CONFIG_PATH})

find_package(CUDA REQUIRED)
find_package(Qt5 REQUIRED Core)

set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})

include(cmake/ProjectFiles.cmake)

file(GLOB_RECURSE src "src/*.cpp" "src/nodes/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/nodes/*.h")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    "include"
                    "RuntimeCompiler"
                    "RuntimeObjectSystem"
                    ${Qt5Core_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}
)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/RuntimeObjectSystem")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/RuntimeCompiler")

set(EagleLib_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)

if(WIN32)
    if(CMAKE_BUILD_TYPE EQUAL "Debug")
        set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_DEBUG_POSTFIX}.lib PARENT_SCOPE)
    else()
        set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_RELEASE_POSTFIX}.lib PARENT_SCOPE)
    endif(CMAKE_BUILD_TYPE EQUAL "Debug")
endif(WIN32)

add_library(RuntimeCompiler
            ${BUILD_TYPE}
            ${RuntimeCompiler_SRCS}
)

add_library(RuntimeObjectSystem
            ${BUILD_TYPE}
            ${RuntimeObjectSystem_SRCS}
)

add_library(EagleLib ${BUILD_TYPE}
	${src} 
	${hdr}
	${RuntimeObjectSystem_SRCS}
)
if(WIN32)
    set_target_properties(EagleLib PROPERTIES COMPILE_FLAGS "/FC")
endif()

TARGET_LINK_LIBRARIES(RuntimeObjectSystem RuntimeCompiler)
TARGET_LINK_LIBRARIES(EagleLib
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        RuntimeCompiler
        RuntimeObjectSystem
)

if(UNIX)
    TARGET_LINK_LIBRARIES(RuntimeObjectSystem dl)
    TARGET_LINK_LIBRARIES(EagleLib dl)
endif()
