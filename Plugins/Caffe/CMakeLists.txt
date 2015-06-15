cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe)


include_directories(
 ${Caffe_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
)

message(STATUS
" ===================== Caffe Plugin ========================
  Caffe found: ${Caffe_FOUND}
  Caffe Include: ${Caffe_INCLUDE_DIRS}

  Caffe libs: ${Caffe_LIBRARIES}

  EagleLib Includes: ${EagleLib_INCLUDE_DIRECTORIES}

  Cuda libs:
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
  ===========================================================")

if(${Caffe_FOUND})

add_library(Caffe SHARED Caffe.h Caffe.cpp)
add_dependencies(Caffe EagleLib)

IF(WIN32)
target_link_libraries(Caffe
    EagleLib
    debug;E:/libsrc/caffe/build_dev/lib/Debug/libcaffe-d.lib;optimized;E:/libsrc/caffe/build_dev/lib/Release/libcaffe.lib
    debug;E:/libsrc/caffe/build_dev/lib/Debug/proto-d.lib;optimized;E:/libsrc/caffe/build_dev/lib/Release/proto.lib
    "E:/code/caffe/3rdparty/lib/gflags.lib"
    "C:/Repo/Raven/packages/ceres-glog.1.10.0/build/native/lib/x64/v120/libglog.lib"
    debug;E:/libsrc/protobuf/vsprojects/x64/debug/libprotobufd.lib;optimized;E:/libsrc/protobuf/vsprojects/x64/release/libprotobuf.lib
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
ELSE(WIN32)
target_link_libraries(Caffe
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

set_target_properties(Caffe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
)
endif()
