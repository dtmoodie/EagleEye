
CMAKE_POLICY(SET CMP0020 NEW)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe)


include_directories(
 ${Caffe_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
)
#STRING(REPLACE ";" "," Caffe_INCLUDE_DIRS_ "${Caffe_INCLUDE_DIRS}")

message(STATUS
" =========== Caffe Plugin ==========
  Caffe found: ${Caffe_FOUND}

  Caffe dir: ${Caffe_CMAKE_DIR}

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

")

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


add_definitions(-DCAFFE_INCLUDES="${Caffe_INCLUDE_DIRS_}")
ENDIF(WIN32)
#SET_TARGET_PROPERTIES(Caffe PROPERTIES LOCATION "${CMAKE_BINARY_DIR}/$(Configuration)/Plugins/libCaffed.lib")
endif()
