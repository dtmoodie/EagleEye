project(MXnet)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(mxnet QUIET)

if(mxnet_FOUND AND TARGET mxnet)
    file(GLOB_RECURSE src "src/*.cpp")
    file(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
    find_package(OpenCV REQUIRED COMPONENTS cudawarping)
    INCLUDE_DIRECTORIES(${mxnet_INCLUDE_DIRS} ${Aquila_INCLUDE_DIRS})
    add_library(MXnet SHARED ${src} ${hdr})

    RCC_LINK_LIB(MXnet
        ${OpenCV_LIBS}
        aquila_metatypes
        aquila_types
        aquila_core
        metaobject_params
        metaobject_object
    )
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        if(mxnet_LIBRARY_DEBUG)
        target_link_libraries(MXnet ${mxnet_LIBRARY_DEBUG})
        elseif(mxnet_LIBRARY_RELEASE)
            target_link_libraries(MXnet ${mxnet_LIBRARY_RELEASE})
        endif()
    else()
        target_link_libraries(MXnet ${mxnet_LIBRARY_RELEASE})
    endif()
    aquila_declare_plugin(MXnet)
    set_target_properties(MXnet PROPERTIES CXX_STANDARD 14)
else(mxnet_FOUND AND TARGET mxnet)
    message(STATUS "-- MXNet NotFound")
endif(mxnet_FOUND AND TARGET mxnet)
