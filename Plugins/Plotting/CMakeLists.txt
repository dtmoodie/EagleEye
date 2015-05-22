cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "include/*.cuh")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB forms "*UIs/*.ui")

find_package(Qt5 REQUIRED Core)
set(Boost_USE_STATIC_LIBS @Boost_USE_STATIC_LIBS@)
set(Boost_USE_STATIC @Boost_USE_STATIC@)
set(Boost_USE_MULTITHREAD @Boost_USE_MULTITHREAD@)
set(BOOST_INCLUDEDIR "@Boost_INCLUDE_DIR@")
set(Boost_NO_BOOST_CMAKE ON)
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
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")



add_library(Plotting SHARED ${src} ${hdr} ${knl} ${UIS} ${MOC})
add_dependencies(Plotting EagleLib)
target_link_libraries(Plotting EagleLib)
