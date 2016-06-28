PROJECT(Caffe)
CMAKE_POLICY(SET CMP0020 NEW)
find_package(Caffe REQUIRED)

find_package(CUDA REQUIRED)
find_package(OpenCV QUIET)

set(CUDNN_DIR "" CACHE PATH "cudnn directory")

find_library(CUDNN_LIBRARY cudnn.lib HINTS "${CUDNN_DIR}/lib/x64/")
find_path(CUDNN_LIB_DIR cudnn.lib HINTS "${CUDNN_DIR}/lib/x64")
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${CUDNN_DIR}/include")
find_path(CUDNN_BIN_DIR cudnn64_4.dll cudnn64_3.dll HINTS "${CUDNN_DIR}/bin")

set_target_properties(
    opencv_core opencv_calib3d opencv_cudaarithm opencv_cudabgsegm opencv_cudacodec opencv_cudafeatures2d 
    opencv_cudafilters opencv_cudaobjdetect opencv_cudaoptflow opencv_cudastereo opencv_cudawarping 
    opencv_cudev opencv_features2d opencv_flann    opencv_highgui opencv_imgcodecs opencv_imgproc 
    opencv_ml opencv_objdetect opencv_photo opencv_shape opencv_stitching opencv_superres 
    opencv_ts opencv_video opencv_videoio opencv_videostab 
    PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

include_directories(
 ${Caffe_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
 ${GLOG_INCLUDE_DIR}
 ${CUDNN_INCLUDE_DIR}
 ${CUDA_TOOLKIT_INCLUDE}
)
SET(CAFFE_BIN_DIR "" CACHE PATH "")
SET(CAFFE_BIN_DIR_DBG ${CAFFE_BIN_DIR} CACHE PATH "" FORCE)
SET(CAFFE_BIN_DIR_OPT ${CAFFE_BIN_DIR} CACHE PATH "" FORCE)

LINK_DIRECTORIES(${LINK_DIRS})
LINK_DIRECTORIES(${CUDNN_LIB_DIR})
file(GLOB_RECURSE src "*.h" "*.cpp")
add_library(Caffe SHARED ${src})
add_dependencies(Caffe EagleLib)
SET(BIN_DIRS "${BIN_DIRS};CAFFE" CACHE STRING "" FORCE)
set_target_properties(libcaffe PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
IF(WIN32)
  RCC_LINK_LIB(Caffe
      EagleLib
      ${CUDA_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_cupti_LIBRARY}
      ${CUDA_curand_LIBRARY}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_npp_LIBRARY}
      ${CUDA_nppc_LIBRARY}
      ${CUDA_nppc_LIBRARY}
      ${CUDA_npps_LIBRARY}
      libcaffe
      ${GLOG_LIBRARY}
      ${CUDNN_LIBRARY}
    )
ELSE(WIN32)
    RCC_LINK_LIB(Caffe
      EagleLib
      ${Caffe_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_cupti_LIBRARY}
      ${CUDA_curand_LIBRARY}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_npp_LIBRARY}
      ${CUDA_nppc_LIBRARY}
      ${CUDA_nppc_LIBRARY}
      ${CUDA_npps_LIBRARY}
    )
ENDIF(WIN32)
INCLUDE(../PluginTemplate.cmake)
