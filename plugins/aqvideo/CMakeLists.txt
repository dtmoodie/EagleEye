project(aqvideo)

file(GLOB_RECURSE src "*.cpp" "*.cu" "*.h" "*.hpp")
ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
    SYSTEM
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
cuda_add_library(aqvideo SHARED ${src})

RCC_LINK_LIB(aqvideo
    ${OpenCV_LIBS}
    ${CUDA_cublas_LIBRARY}
    metaobject_core metaobject_object metaobject_params metaobject_metaparams aquila_core
)

aquila_declare_plugin(aqvideo)
set_plugin_available(aqvideo)
