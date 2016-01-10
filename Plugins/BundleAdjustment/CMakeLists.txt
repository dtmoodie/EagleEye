
CMAKE_POLICY(SET CMP0020 NEW)
find_package(CUDA REQUIRED)

find_package(Ceres QUIET)
IF(WIN32)
	set(Boost_USE_STATIC_LIBS        ON)
	set(Boost_USE_MULTITHREADED      ON)
	set(Boost_USE_STATIC_RUNTIME     OFF)
ELSE(WIN32)
	set(Boost_USE_STATIC_LIBS        OFF)
	set(Boost_USE_MULTITHREADED      ON)
	set(Boost_USE_STATIC_RUNTIME     OFF)
ENDIF(WIN32)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono regex)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES}) 

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

  ${CUDA_LIBRARIES}

  ${EagleLib_LIBRARIES}
")
LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
STRING(REGEX REPLACE ";" "+" CERES_INCLUDE_DIRS_ "${CERES_INCLUDE_DIRS}+" )
add_definitions(-DCERES_LIB_DIR="${Ceres_DIR}/../lib")
add_definitions(-DCERES_INC_DIR="${CERES_INCLUDE_DIRS_}")
add_library(BundleAdjustment SHARED BundleAdjustment.h BundleAdjustment.cpp)
target_link_libraries(BundleAdjustment ${Boost_LIBRARIES} ${EagleLib_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(BundleAdjustment PROPERTIES FOLDER "Plugins")
add_dependencies(BundleAdjustment EagleLib)
ELSE(${Ceres_FOUND})

message(STATUS "Ceres not found")
ENDIF(${Ceres_FOUND})
