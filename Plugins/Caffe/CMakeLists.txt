PROJECT(Caffe)
CMAKE_POLICY(SET CMP0020 NEW)
find_package(Caffe REQUIRED)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(
 ${Caffe_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
 ${GLOG_INCLUDE_DIR}
)
SET(CAFFE_BIN_DIR "" CACHE PATH "")
SET(CAFFE_BIN_DIR_DBG ${CAFFE_BIN_DIR} CACHE PATH "" FORCE)
SET(CAFFE_BIN_DIR_OPT ${CAFFE_BIN_DIR} CACHE PATH "" FORCE)

LINK_DIRECTORIES(${LINK_DIRS})
add_library(Caffe SHARED Caffe.h Caffe.cpp)
add_dependencies(Caffe EagleLib)
SET(BIN_DIRS "${BIN_DIRS};CAFFE" CACHE STRING "" FORCE)

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
	  ${Caffe_LIBRARIES}
	  ${GLOG_LIBRARY}
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
