cmake_minimum_required(VERSION 2.8)

project(Tests)
#if(WIN32)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FC")
#else(WIN32)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fPIC")
#endif(WIN32)

set(Boost_USE_STATIC_LIBS @Boost_USE_STATIC_LIBS@)
set(Boost_USE_STATIC @Boost_USE_STATIC@)
set(Boost_USE_MULTITHREAD @Boost_USE_MULTITHREAD@)
set(BOOST_INCLUDEDIR "@Boost_INCLUDE_DIR@")
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)


INCLUDE_DIRECTORIES(
	${EagleLib_INCLUDE_DIRECTORIES}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(test_shared_ptr test_shared_ptr.cpp)
add_dependencies(test_shared_ptr EagleLib)
target_link_libraries(test_shared_ptr EagleLib)

add_executable(test_tree test_tree.cpp)
add_dependencies(test_tree EagleLib)
target_link_libraries(test_tree EagleLib)

add_executable(test_rcc test_rcc.cpp)
add_dependencies(test_rcc EagleLib)
target_link_libraries(test_rcc EagleLib)
