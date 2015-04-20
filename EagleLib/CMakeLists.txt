cmake_minimum_required(VERSION 2.8)
project(EagleLib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(OPENCV_STATIC OFF)
find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
find_package(CUDA REQUIRED)
find_package(Qt5 REQUIRED Core)

IF(WIN32)
	set(Boost_USE_STATIC_LIBS        ON)
	set(Boost_USE_MULTITHREADED      ON)
	set(Boost_USE_STATIC_RUNTIME     OFF)
ELSE(WIN32)
	set(Boost_USE_STATIC_LIBS        OFF)
	set(Boost_USE_MULTITHREADED      ON)
	set(Boost_USE_STATIC_RUNTIME     OFF)
ENDIF(WIN32)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})

include(cmake/ProjectFiles.cmake)

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "include/*.cuh")
file(GLOB_RECURSE knl "src/*.cu")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    "include"
                    "RuntimeCompiler"
                    "RuntimeObjectSystem"
                    ${Qt5Core_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    "${CMAKE_CURRENT_SOURCE_DIR}/RuntimeCompiler"
                    "${CMAKE_CURRENT_SOURCE_DIR}/RuntimeObjectSystem"
)



# for RCC to be able to recompile modules
LIST(GET OpenCV_INCLUDE_DIRS 0 OpenCV_INCLUDE_DIR)
LIST(GET OpenCV_LIB_DIR 0 CV_LIB_DIR)
if(${CV_LIB_DIR} MATCHES NOTFOUND)
message("OpenCV_LIB_DIR not set, attempting to find library dir")
endif()


ADD_DEFINITIONS( -DBOOST_INCLUDES="${Boost_INCLUDE_DIRS}"   )
ADD_DEFINITIONS( -DBOOST_LIB_DIR="${Boost_LIBRARY_DIRS}")

ADD_DEFINITIONS( -DOPENCV_INCLUDES="${OpenCV_INCLUDE_DIR}")
ADD_DEFINITIONS( -DOPENCV_LIB_DIR="${CV_LIB_DIR}")
ADD_DEFINITIONS( -DCUDA_INCLUDES="${CUDA_INCLUDE_DIRS}" )
ADD_DEFINITIONS( -DCUDA_LIB_DIR="${CUDA_TOOLKIT_ROOT_DIR}/lib64")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

# Reporting section
message(
"================ EagleLib ===============
    Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}
    OpenCV found at: ${OpenCV_CONFIG_PATH}
    NVCC Flags: ${CUDA_NVCC_FLAGS}
++++++++++ RCC++ Defines ++++++++++++++++++
    OpenCV include dir: ${OpenCV_INCLUDE_DIR}
    Boost library dir: ${Boost_LIBRARY_DIRS}
    OpenCV library dir: ${CV_LIB_DIR}
    Cuda include dir: ${CUDA_INCLUDE_DIRS}
    Cuda RT libs: ${CUDA_LIBRARIES}
    Cuda root dir: ${CUDA_TOOLKIT_ROOT_DIR}
************* FILES ***********************
    Headers: ${hdr}
    Source files: ${src}
    Kernels: ${knl}
=====================================")


set(EagleLib_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)

if(WIN32)
    if(CMAKE_BUILD_TYPE EQUAL "Debug")
        set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_DEBUG_POSTFIX}.lib PARENT_SCOPE)
    else()
        set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_RELEASE_POSTFIX}.lib PARENT_SCOPE)
    endif(CMAKE_BUILD_TYPE EQUAL "Debug")
endif(WIN32)

add_library(RuntimeCompiler
            STATIC
            ${RuntimeCompiler_SRCS}
)

add_library(RuntimeObjectSystem
            STATIC
            ${RuntimeObjectSystem_SRCS}
)

cuda_add_library(EagleLib SHARED
	${src} 
	${hdr}
        ${knl}
	${RuntimeObjectSystem_SRCS}
)

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
