set(DARKNET_DIR "" CACHE PATH "Path to root install of darknet")

if(WIN32)
    find_library(DARKNET_LIBRARY yolo_cpp_dll.lib
            HINTS ${DARKNET_DIR}/lib
    )

    find_path(DARKNET_INCLUDE yolo_v2_class.hpp
        HINTS ${DARKNET_DIR}/include
    )

else(WIN32)
    find_library(DARKNET_LIBRARY darknet
        HINTS ${DARKNET_DIR}
              ${DARKNET_DIR}/lib
    )
    find_path(DARKNET_INCLUDE darknet.h
        HINTS ${DARKNET_DIR}/include
    )
endif(WIN32)

if(DARKNET_LIBRARY AND DARKNET_INCLUDE)
    find_package(CUDNN QUIET)
    if(CUDNN_FOUND)
        find_package(OpenCV QUIET COMPONENTS cudawarping cudaimgproc)
        file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp")
        INCLUDE_DIRECTORIES(
            ${DARKNET_INCLUDE}
            ${OpenCV_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${MetaObject_INCLUDE_DIRS}
        )
        cuda_add_library(aqdarknet SHARED ${src})
        target_link_libraries(aqdarknet
            aquila_core
            aquila_types
            metaobject_metaparams
            metaobject_params
            ${DARKNET_LIBRARY}
            aqcore
            ${CUDNN_LIBRARY}
        )
        aquila_declare_plugin(aqdarknet)
        set_plugin_available(aqdarknet)
        set_plugin_status(aqdarknet "Darknet ${DARKNET_DIR}")
        aq_install_python_test(scripts/test_yolo.py)
        aquila_install_dependent_lib(${DARKNET_LIBRARY})
    else()
        set_plugin_unavailable(aqdarknet)
        set_plugin_status(aqdarknet "cudnn not found")
    endif()

else()
    set_plugin_unavailable(aqdarknet)
    set_plugin_status(aqdarknet "Darknet not found")
endif()
