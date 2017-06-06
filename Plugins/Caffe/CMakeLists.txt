PROJECT(Caffe)
CMAKE_POLICY(SET CMP0020 NEW)
find_package(Caffe QUIET)
set(Plugin_Caffe_available "FALSE")
if(Caffe_FOUND)
    IF(UNIX)
      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
      set(CUDA_NVCC_FLAGS "-std=c++11;--expt-relaxed-constexpr;${CUDA_NVCC_FLAGS}")
    ENDIF()
  set(Plugin_Caffe_available "TRUE")
  find_package(OpenCV QUIET)

  set(CUDNN_DIR "" CACHE PATH "cudnn directory")

  find_library(CUDNN_LIBRARY cudnn.lib cudnn.so libcudnn.so cudnn HINTS "${CUDNN_DIR}/lib/x64/" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  find_path(CUDNN_LIB_DIR cudnn.lib cudnn.so libcudnn.so cudnn HINTS "${CUDNN_DIR}/lib/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${CUDNN_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/include")
  find_path(CUDNN_BIN_DIR cudnn64_4.dll cudnn64_3.dll libcudnn.so cudnn cudnn.so HINTS "${CUDNN_DIR}/bin" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
    include_directories(
      ${Caffe_INCLUDE_DIRS}
      ${Aquila_INCLUDE_DIRECTORIES}
      ${GLOG_INCLUDE_DIR}
      ${CUDNN_INCLUDE_DIR}
      ${CUDA_TOOLKIT_INCLUDE}
    )
    add_definitions(-DUSE_CUDNN)

    LINK_DIRECTORIES(${LINK_DIRS})
    file(GLOB_RECURSE src "*.h" "*.hpp" "*.cpp" "*.cu")
    cuda_add_library(Caffe SHARED ${src})

    RCC_LINK_LIB(Caffe
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
          ${GLOG_LIBRARY}
          ${CUDNN_LIBRARY}
          aquila_types
          aquila_metatypes
          aquila_core
          aquila_serialization
    )

    IF(WIN32)
        if(TARGET libcaffe)
            get_target_property(CAFFE_BIN libcaffe IMPORTED_LOCATION_DEBUG)
        endif(TARGET libcaffe)
        get_filename_component(CAFFE_BIN_DIR_ "${CAFFE_BIN}" DIRECTORY)
        set(CAFFE_BIN_DIR ${CAFFE_BIN_DIR_} CACHE PATH "" FORCE)
        SET(CAFFE_BIN_DIR_DBG ${CAFFE_BIN_DIR} CACHE PATH "" FORCE)
        SET(CAFFE_BIN_DIR_OPT ${CAFFE_BIN_DIR} CACHE PATH "" FORCE)
        SET(BIN_DIRS "${BIN_DIRS};CAFFE;CUDNN" CACHE STRING "" FORCE)
        set_target_properties(libcaffe PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
        RCC_LINK_LIB(Caffe
          libcaffe
        )
    ELSE(WIN32)
        RCC_LINK_LIB(Caffe
            caffe
        )
    ENDIF(WIN32)
    #INCLUDE(../PluginTemplate.cmake)
    aquila_declare_plugin(Caffe)

  else()
    message("CUDNN not found")
    message(STATUS
        " CUDNN lib: ${CUDNN_LIBRARY}\n"
        "  include:  ${CUDNN_INCLUDE_DIR}\n"
        "  lib dir:  ${CUDNN_LIB_DIR}\n"
        "  bin dir:  ${CUDNN_BIN_DIR}\n"
    )
  endif()
else()
  message("Caffe not found")
endif()
