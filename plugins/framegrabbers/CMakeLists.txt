PROJECT(aqframegrabbers)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})
  link_directories(${BOOST_LIBRARYDIR})
  link_directories(${Boost_LIBRARY_DIR_DEBUG})
  link_directories(${Boost_LIBRARY_DIR_RELEASE})
  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV 3.0 QUIET COMPONENTS core imgcodecs cudacodec cudaarithm)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if(WIN32)
    FIND_PACKAGE(GStreamerWindows QUIET)
else(WIN32)
    FIND_PACKAGE(Gstreamer QUIET)
    INCLUDE_DIRECTORIES("/usr/lib/x86_64-linux-gnu/glib-2.0/include")
    INCLUDE_DIRECTORIES("/usr/lib/arm-linux-gnueabihf/glib-2.0/include")
    INCLUDE_DIRECTORIES("/usr/include/glib-2.0")
endif(WIN32)
if(GSTREAMER_FOUND)
    INCLUDE_DIRECTORIES(${GSTREAMER_gst_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${GSTREAMER_gstconfig_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${GSTREAMER_glibconfig_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${GSTREAMER_glib_INCLUDE_DIR})

    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_gst_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_gstconfig_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_glibconfig_INCLUDE_DIR})
    LIST(APPEND PROJECT_INCLUDES ${GSTREAMER_glib_INCLUDE_DIR})
    add_definitions(-DHAVE_GSTREAMER)
endif()

ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")

cuda_add_library(aqframegrabbers SHARED ${src} ${hdr} ${knl})

RCC_LINK_LIB(aqframegrabbers
        aquila_core
        metaobject_core metaobject_object metaobject_params metaobject_metaparams
        ${OpenCV_LIBS}
        ${CUDA_CUBLAS_LIBRARIES}
        aquila_types
        aquila_metatypes
)
if(GSTREAMER_FOUND)
    RCC_LINK_LIB(aqframegrabbers
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
endif()

# ------------- tests
if(BUILD_TESTS)
    add_subdirectory("tests")
endif()

aquila_declare_plugin(aqframegrabbers)
set_plugin_available(aqframegrabbers)

set(FRAMEGRABBERS_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)

aq_install_python_test(tests/test_directory_grabber.py)
