PROJECT(SLAM)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
    CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)

set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
  ${INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/dtam/Cpp
)
find_package(Eigen3 REQUIRED)
  include_directories(${EIGEN_INCLUDE_DIRS})
if(EIGEN_INCLUDE_DIRS)
  find_package(G2O QUIET)
  if(G2O_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    ADD_SUBDIRECTORY("lsd_slam/lsd_slam_core")

    file(GLOB_RECURSE src "src/*.cpp")
    file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp" "src/*.cuh")

    IF(WIN32)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    ENDIF()

    cuda_add_library(SLAM SHARED ${src} ${hdr} ${knl})

    add_dependencies(SLAM Aquila)
    add_dependencies(SLAM OpenDTAM)
    RCC_LINK_LIB(SLAM
      ${OpenCV_LIBS}
      ${CUDA_CUBLAS_LIBRARIES}
      Aquila
      OpenDTAM
    )

    INCLUDE(../PluginTemplate.cmake)
  endif(G2O_FOUND)
endif()
