cmake_minimum_required(VERSION 2.8)

project(Plugins)


find_package(Qt5 REQUIRED Core)
INCLUDE_DIRECTORIES(${Qt5Core_INCLUDE_DIRS})

add_subdirectory(EagleEye_test_plugin)


