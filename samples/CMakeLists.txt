cmake_minimum_required(VERSION 2.8)

project(samples)

INCLUDE_DIRECTORIES(${EagleLib_INCLUDE_DIRECTORIES})

add_executable(SimpleConsole SimpleConsole.cpp TestNode.cpp)
add_dependencies(SimpleConsole EagleLib)
target_link_libraries(SimpleConsole EagleLib)
