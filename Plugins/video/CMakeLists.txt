project(video)

file(GLOB_RECURSE src "*.cpp" "*.cu" "*.h" "*.hpp")
ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
cuda_add_library(video SHARED ${src})

RCC_LINK_LIB(video
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
    metaobject_core metaobject_object metaobject_params metaobject_metaparams aquila_core
)

#INCLUDE(../PluginTemplate.cmake)
aquila_declare_plugin(video)
