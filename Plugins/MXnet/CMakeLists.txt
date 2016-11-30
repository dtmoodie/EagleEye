list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
project(MXnet)
find_package(mxnet QUIET)

if(MXNET_FOUND)
  find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
  find_package(CUDA REQUIRED)
    include_directories(${CUDA_TOOLKIT_INCLUDE})
  rcc_find_library(CUDNN_LIBRARY cudnn.lib cudnn.so libcudnn.so cudnn HINTS "${CUDNN_DIR}/lib/x64/" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  rcc_find_path(CUDNN_LIB_DIR cudnn.lib cudnn.so libcudnn.so cudnn HINTS "${CUDNN_DIR}/lib/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  rcc_find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${CUDNN_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/include")
  rcc_find_path(CUDNN_BIN_DIR cudnn64_4.dll cudnn64_3.dll libcudnn.so cudnn cudnn.so HINTS "${CUDNN_DIR}/bin" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  LINK_DIRECTORIES(${LINK_DIRS})
  
  include_directories(${MXNET_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
  file(GLOB_RECURSE src "src/*.cpp")
  file(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
  add_library(MXnet SHARED ${src} ${hdr})
  RCC_LINK_LIB(MXnet ${MXNET_LIBRARY}
					 ${OpenBLAS_LIB}
					 ${CUDNN_LIBRARY}
					 EagleLib
					 MetaObject)
  INCLUDE(../PluginTemplate.cmake)
endif()