cmake_minimum_required(VERSION 2.8)

project(EagleEye)
if(${CMAKE_BUILD_TYPE})
message(${CMAKE_BUILD_TYPE})
else()
#SET(CMAKE_BUILD_TYPE "Debug")
endif(${CMAKE_BUILD_TYPE})
#set(CMAKE_DEBUG_POSTFIX "d")
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/Bin CACHE PATH "Library output path")
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/Bin CACHE PATH "Executable output path")
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release")

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
message("Debug Mode")
SET(CMAKE_CXX_FLAGS "-O0 -fno-builtin")
SET(CMAKE_CC_FLAGS "-O0 -fno-builtin")
SET(CUDA_NVCC_FLAGS -O0 --compiler-options -fpermissive -arch sm_30 --maxrregcount 64 --ptxas-options=-v)
else()
SET(CMAKE_CXX_FLAGS "-O3 -fno-builtin")
SET(CMAKE_CC_FLAGS "-O3 -fno-builtin")
SET(CUDA_NVCC_FLAGS -O3 --compiler-options -fpermissive -arch sm_30 --maxrregcount 64 --ptxas-options=-v)
ENDIF(CMAKE_BUILD_TYPE MATCHES "Debug")





IF(WIN32)
    add_definitions(-DWIN32)
    add_definitions(-D_WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    message("Windows")
ELSE(WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message("Linux")
ENDIF(WIN32)


message(Build type: ${CMAKE_BUILD_TYPE})
add_subdirectory(EagleLib)
add_subdirectory(EagleEye)
