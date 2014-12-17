cmake_minimum_required(VERSION 2.8)
project(EagleLib)

find_package(OpenCV REQUIRED core imgproc highgui cuda cudaimgproc cudawarping)

find_package(CUDA REQUIRED)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})

file(GLOB_RECURSE src "src/*.cpp" "src/nodes/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/nodes/*.h")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("include")
set(EagleLib_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
set(EagleLib_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libEagleLib.so PARENT_SCOPE)

foreach(lib ${OpenCV_LIBS})
message(${lib})
endforeach(lib)


cuda_add_library(EagleLib SHARED ${src} ${hdr})

TARGET_LINK_LIBRARIES(EagleLib ${Boost_LIBRARIES} ${OpenCV_LIBS} )
