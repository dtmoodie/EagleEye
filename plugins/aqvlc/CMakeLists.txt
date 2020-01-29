PROJECT(VLC)
CMAKE_POLICY(SET CMP0020 NEW)
FIND_PACKAGE(LibVLC QUIET)
IF(LIBVLC_FOUND)
    FIND_PACKAGE(CUDA REQUIRED)
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
    FIND_PACKAGE(Qt5Network)
    INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${INCLUDE_DIRS}
        ${LIBVLC_INCLUDE_DIR}
        ${INCLUDE_DIRS}
    )

    LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    LINK_DIRECTORIES(${LINK_DIRS})

    FILE(GLOB_RECURSE src "src/*.cpp")
    FILE(GLOB_RECURSE hdr "include/*.h")

    IF(LIBVLC_LIBRARY)
        CUDA_ADD_LIBRARY(aqvlc SHARED ${src} ${hdr})
        RCC_LINK_LIB(aqvlc
            ${OpenCV_LIBS}
            aquila_core
            aquila_types
            metaobject_core
            metaobject_object
            metaobject_params
            metaobject_metaparams
        )
        TARGET_LINK_LIBRARIES(aqvlc
            ${LINK_LIBS}
            ${LIBVLC_LIBRARY} ${Boost_LIBRARIES}
            ${LIBVLCCORE_LIBRARY}
        )
        aquila_declare_plugin(aqvlc)
    ENDIF()
else()
    set_plugin_status(aqvlc "VLC lib not found")
    set_plugin_unavailable(aqvlc)
ENDIF(LIBVLC_FOUND)
