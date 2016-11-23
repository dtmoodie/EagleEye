PROJECT(GStreamer)
CMAKE_POLICY(SET CMP0020 NEW)
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/CMake")
include(CMake)
FIND_PACKAGE(CUDA REQUIRED)
SET(PROJECT_INCLUDES)
ADD_DEFINITIONS(${DEFS})
IF(WIN32)
    FIND_PACKAGE(GStreamerWindows QUIET)
ELSE(WIN32)
    FIND_PACKAGE(Gstreamer QUIET)
ENDIF(WIN32)

IF(GSTREAMER_FOUND)
    if(WIN32)

    else()
        FIND_PACKAGE(PkgConfig)
        pkg_search_module(GLIB2 glib-2.0)
            INCLUDE_DIRECTORIES(${GSTREAMER_gst_INCLUDE_DIR})
            INCLUDE_DIRECTORIES(${GSTREAMER_gstconfig_INCLUDE_DIR})
            INCLUDE_DIRECTORIES(${Glib_INCLUDE_DIR})

        if(GLIB2_FOUND)
            INCLUDE_DIRECTORIES("/usr/lib/x86_64-linux-gnu/glib-2.0/include")
            INCLUDE_DIRECTORIES("/usr/lib/arm-linux-gnueabihf/glib-2.0/include")
            INCLUDE_DIRECTORIES("/usr/include/glib-2.0")
        else(GLIB2_FOUND)
            message(STATUS "glib2 not found")
        endif(GLIB2_FOUND)
        LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib)
    endif()

    INCLUDE_DIRECTORIES(${GSTREAMER_gst_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${GSTREAMER_gstconfig_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${GSTREAMER_glibconfig_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${GSTREAMER_glib_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_gst_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_gstconfig_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_glibconfig_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_glib_INCLUDE_DIR})
    FIND_PACKAGE(Qt5 REQUIRED Network)
    FIND_PACKAGE(OpenCV REQUIRED core)
    FILE(GLOB_RECURSE src "src/*.cpp")
    FILE(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
    INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
    LINK_DIRECTORIES(${LINK_DIRS})
    STRING(REGEX REPLACE ";" "+" PROJECT_INCLUDES "${PROJECT_INCLUDES}" )
    ADD_DEFINITIONS(-DPROJECT_INCLUDES="${PROJECT_INCLUDES}")
    GET_FILENAME_COMPONENT(LIB_DIR "${GSTREAMER_gstapp_LIBRARY}" DIRECTORY)
    IF(${GSTREAMER_gstrtspserver_LIBRARY})
        add_definitions(-DHAVE_GST_RTSPSERVER)
    endif()
    ADD_DEFINITIONS(-DPROJECT_LIB_DIRS="${LIB_DIR}")
    ADD_LIBRARY(GStreamer SHARED ${src} ${hdr})
    RCC_LINK_LIB(GStreamer
        ${OpenCV_LIBS}
        EagleLib
        Qt5::Network
        ${GSTREAMER_gstapp_LIBRARY}
        ${GSTREAMER_gstaudio_LIBRARY}
        ${GSTREAMER_gstbase_LIBRARY}
        ${GSTREAMER_gstcontroller_LIBRARY}
        ${GSTREAMER_gstnet_LIBRARY}
        ${GSTREAMER_gstpbutils_LIBRARY}
        ${GSTREAMER_gstreamer_LIBRARY}
        ${GSTREAMER_gstriff_LIBRARY}
        ${GSTREAMER_gstrtp_LIBRARY}
        ${GSTREAMER_gstrtsp_LIBRARY}
        ${GSTREAMER_gstsdp_LIBRARY}
        ${GSTREAMER_gsttag_LIBRARY}
        ${GSTREAMER_gstvideo_LIBRARY}
        ${Glib_LIBRARY}
        ${GLIB_LIBRARY}
        ${GOBJECT_LIBRARY}
    )
    IF(${GSTREAMER_gstrtspserver_LIBRARY})
        RCC_LINK_LIB(Gstreamer ${GSTREAMER_gstrtspserver_LIBRARY})
    ENDIF()

    INCLUDE(../PluginTemplate.cmake)
    ADD_DEPENDENCIES(GStreamer EagleLib)
else()
    message("Gstreamer not found")
ENDIF()
