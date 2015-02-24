cmake_minimum_required(VERSION 2.8)
project(EagleLib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if(UNIX)
        set(BUILD_TYPE STATIC)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
#if(WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FC")
#else(WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fPIC")
#endif(WIN32)
endif()

else()
        set(BUILD_TYPE STATIC)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

set(OPENCV_STATIC ON)
set(OpenCV_ROOT "/usr/local/share/OpenCV")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/share/OpenCV)
find_package(OpenCV 3.0 REQUIRED core imgproc highgui cuda cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
message(${OpenCV_CONFIG_PATH})

find_package(CUDA REQUIRED)
find_package(Qt5 REQUIRED Core)
IF(QT5_FOUND)
    message(Found Qt5)

ENDIF()

set(Boost_USE_STATIC_LIBS @Boost_USE_STATIC_LIBS@)
set(Boost_USE_STATIC @Boost_USE_STATIC@)
set(Boost_USE_MULTITHREAD @Boost_USE_MULTITHREAD@)
set(BOOST_INCLUDEDIR "@Boost_INCLUDE_DIR@")
set(Boost_NO_BOOST_CMAKE ON)
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

#SET(EagleLib_DEFINITIONS "-DEAGLE_LIB_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

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
if(MSVC)
		set_target_properties(EagleLib PROPERTIES COMPILE_FLAGS "/FC")
	endif()

foreach(lib ${OpenCV_LIBS})
message(${lib})
endforeach(lib)

TARGET_LINK_LIBRARIES(EagleLib
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        RuntimeCompiler
        RuntimeObjectSystem
)
if(UNIX)
TARGET_LINK_LIBRARIES(EagleLib 
        dl
)
endif(UNIX)
