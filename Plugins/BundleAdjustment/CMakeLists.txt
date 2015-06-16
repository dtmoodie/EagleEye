cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

find_package(Ceres)

IF(Ceres_FOUND)
include_directories(
 ${Ceres_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
)

add_library(Ceres SHARED BundleAdjustment.h BundleAdjustment.cpp)
ENDIF(Ceres_FOUND)
