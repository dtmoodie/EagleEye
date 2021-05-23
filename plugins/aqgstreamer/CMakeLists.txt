PROJECT(aqgstreamer)
CMAKE_POLICY(SET CMP0020 NEW)
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/CMake")
include(CMake)
FIND_PACKAGE(CUDA REQUIRED)
SET(PROJECT_INCLUDES)
ADD_DEFINITIONS(${DEFS})
IF(WIN32)
    FIND_PACKAGE(GStreamerWindows QUIET)
ELSE(WIN32)
    FIND_PACKAGE(Gstreamer REQUIRED)
ENDIF(WIN32)

IF(NOT GSTREAMER_FOUND)
    set_plugin_unavailable(aqgstreamer)
    set_plugin_status(aqgstreamer "gstreamer not found")
    return()
endif(NOT GSTREAMER_FOUND)

FIND_PACKAGE(PkgConfig)
pkg_search_module(GLIB2 glib-2.0)

if(NOT GLIB2_FOUND)
    set_plugin_unavailable(aqgstreamer)
    set_plugin_status(aqgstreamer "glib2 not found")
    return()
endif(NOT GLIB2_FOUND)

FIND_PACKAGE(OpenCV QUIET COMPONENTS core imgcodecs)
if(NOT OpenCV_FOUND)
    set_plugin_status(aqgstreamer "OpenCV core and imgcodecs not found")
    set_plugin_unavailable(aqgstreamer)
    return()
endif()


FILE(GLOB_RECURSE src "src/*.cpp")
FILE(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
add_library(aqgstreamer SHARED ${src} ${hdr})

target_include_directories(aqgstreamer
    PUBLIC
        ${GSTREAMER_gst_INCLUDE_DIR}
        ${GSTREAMER_gstconfig_INCLUDE_DIR}
        ${GSTREAMER_glibconfig_INCLUDE_DIR}
        ${GSTREAMER_glib_INCLUDE_DIR}
        "/usr/lib/x86_64-linux-gnu/glib-2.0/include"
        "/usr/lib/arm-linux-gnueabihf/glib-2.0/include"
        "/usr/include/glib-2.0"
)

target_link_libraries(aqgstreamer
    ${OpenCV_LIBS}

    ${GSTREAMER_gstreamer_LIBRARY}
    ${GSTREAMER_gstapp_LIBRARY}
    ${GSTREAMER_gstbase_LIBRARY}
    ${Glib_LIBRARY}
    ${GLIB_LIBRARY}
    ${GLIB2_LIBRARIES}

    ${GOBJECT_LIBRARY}
    metaobject_core
    metaobject_object
    metaobject_params
    metaobject_metaparams

    aquila_core
    aquila_types
    aquila_utilities
)

GET_FILENAME_COMPONENT(LIB_DIR "${GSTREAMER_gstapp_LIBRARY}" DIRECTORY)
IF(${GSTREAMER_gstrtspserver_LIBRARY})
    target_compile_definitions(aqgstreamer -DHAVE_GST_RTSPSERVER)
    target_link_libraries(aqgstreamer ${GSTREAMER_gstrtspserver_LIBRARY})
endif()
metaobject_declare_plugin(aqgstreamer)
set_plugin_available(aqgstreamer)
set_plugin_status(aqgstreamer "gstreamer include: ${GSTREAMER_gst_INCLUDE_DIR} library: ${GSTREAMER_gstreamer_LIBRARY}")
