cmake_minimum_required(VERSION 2.8)

project(Plugins)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/Bin/Plugins CACHE PATH "Library output path")
add_subdirectory(PCL_bridge)
add_subdirectory(Plotting)


