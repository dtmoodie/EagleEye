
CMAKE_POLICY(SET CMP0020 NEW)
find_package(CUDA REQUIRED)

find_package(Ceres QUIET)

IF(Ceres_FOUND)
include_directories(
 ${Ceres_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
)

add_library(Ceres SHARED BundleAdjustment.h BundleAdjustment.cpp)
ENDIF(Ceres_FOUND)
