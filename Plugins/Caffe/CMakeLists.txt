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
  ===========================================================")

if(${Caffe_FOUND})

add_library(Caffe SHARED Caffe.h Caffe.cpp)
add_dependencies(Caffe EagleLib)
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

set_target_properties(Caffe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
)
endif()
