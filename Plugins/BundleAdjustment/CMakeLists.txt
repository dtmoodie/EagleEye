
CMAKE_POLICY(SET CMP0020 NEW)
find_package(CUDA REQUIRED)

find_package(Ceres QUIET)

IF(${Ceres_FOUND})
include_directories(
 ${CERES_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
 
)
message( STATUS
"========== Ceres ================
  Include: ${CERES_INCLUDE_DIRS}
  
  Libs: ${CERES_LIBRARIES}
  
  Lib dir: ${CERES_LIBRARY_DIR}
")
STRING(REGEX REPLACE ";" "+" CERES_INCLUDE_DIRS_ "${CERES_INCLUDE_DIRS}+" )
add_definitions(-DCERES_LIB_DIR="${Ceres_DIR}/../lib")
add_definitions(-DCERES_INC_DIR="${CERES_INCLUDE_DIRS_}")
add_library(Ceres SHARED BundleAdjustment.h BundleAdjustment.cpp)
ELSE(${Ceres_FOUND})

message(STATUS "Ceres not found")
ENDIF(${Ceres_FOUND})
