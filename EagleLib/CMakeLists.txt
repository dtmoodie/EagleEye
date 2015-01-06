cmake_minimum_required(VERSION 2.8)
project(EagleLib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if(UNIX)
        set(BUILD_TYPE STATIC)
else()
        set(BUILD_TYPE STATIC)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

set(OPENCV_STATIC ON)
find_package(OpenCV REQUIRED core imgproc highgui cuda cudaimgproc cudawarping cudafeatures2d)
message(${OpenCV_CONFIG_PATH})

find_package(CUDA REQUIRED)

set(Boost_USE_STATIC_LIBS @Boost_USE_STATIC_LIBS@)
set(Boost_USE_STATIC @Boost_USE_STATIC@)
set(Boost_USE_MULTITHREAD @Boost_USE_MULTITHREAD@)
set(BOOST_INCLUDEDIR "@Boost_INCLUDE_DIR@")
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


file(GLOB_RECURSE src "src/*.cpp" "src/nodes/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/nodes/*.h")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    "include"
                    "RuntimeCompiler"
                    "RuntimeObjectSystem"
)
include(cmake/ProjectFiles.cmake)
set(EagleLib_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)

if(WIN32)
	if(CMAKE_BUILD_TYPE EQUAL "Debug")
		set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_DEBUG_POSTFIX}.lib PARENT_SCOPE)
		message(${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_DEBUG_POSTFIX}.dll)
	else()
		set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_RELEASE_POSTFIX}.lib PARENT_SCOPE)
		message(${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE}/EagleLib${CMAKE_RELEASE_POSTFIX}.dll)
	endif(CMAKE_BUILD_TYPE EQUAL "Debug")
else()
	if(CMAKE_BUILD_TYPE EQUAL "Debug")
                set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/libEagleLib${CMAKE_DEBUG_POSTFIX}.a PARENT_SCOPE)
		else()
                set(EagleLib_LIBRARIES ${CMAKE_BINARY_DIR}/Bin/libEagleLib${CMAKE_RELEASE_POSTFIX}.a PARENT_SCOPE)
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

target_link_libraries(RuntimeObjectSystem RuntimeCompiler)
if(UNIX)
        target_link_libraries(RuntimeObjectSystem dl)
endif()



add_library(EagleLib STATIC
	${src} 
	${hdr}
	${RuntimeObjectSystem_SRCS}
)

foreach(lib ${OpenCV_LIBS})
message(${lib})
endforeach(lib)

get_property(dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)

message("Include directories: ")
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
message("End of include directories")

if(UNIX)
TARGET_LINK_LIBRARIES(EagleLib 
	${Boost_LIBRARIES} 
	${OpenCV_LIBS} 
        RuntimeCompiler
        RuntimeObjectSystem
        dl
)
else(UNIX)
TARGET_LINK_LIBRARIES(EagleLib
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        RuntimeCompiler
        RuntimeObjectSystem
)

endif(UNIX)
