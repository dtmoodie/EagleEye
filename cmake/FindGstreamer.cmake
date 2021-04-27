FILE(TO_CMAKE_PATH "$ENV{GSTREAMER_DIR}" TRY1_DIR)
FILE(TO_CMAKE_PATH "${GSTREAMER_DIR}" TRY2_DIR)
FILE(GLOB GSTREAMER_DIR ${TRY1_DIR} ${TRY2_DIR})


set(gstreamer_include_search_paths
    ${GSTREAMER_DIR}/include
    ${GSTREAMER_DIR}/lib/include
    /usr/local/include/gstreamer-1.0
    /usr/include/gstreamer-1.0
    /usr/local/lib/include/gstreamer-1.0
    /usr/lib/include/gstreamer-1.0
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/
)

set(gstreamer_library_search_paths
    ${GSTREAMER_DIR}/bin
    ${GSTREAMER_DIR}/win32/bin
    ${GSTREAMER_DIR}/bin/bin
    C:/gstreamer/bin
    ${GSTREAMER_DIR}/lib
    ${GSTREAMER_DIR}/win32/lib
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

find_path(GSTREAMER_gst_INCLUDE_DIR gst/gst.h
    PATHS ${gstreamer_include_search_paths}
    ENV INCLUDE DOC "Directory containing gst/gst.h include file"
)



find_path(GSTREAMER_gstconfig_INCLUDE_DIR gst/gstconfig.h
    PATHS ${gstreamer_include_search_paths}
    ENV INCLUDE
    DOC "Directory containing gst/gstconfig.h include file"
)


find_library(GSTREAMER_gstaudio_LIBRARY
    NAMES
        gstaudio-1.0
        libgstaudio-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstaudio library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)


find_library(GSTREAMER_gstapp_LIBRARY
    NAMES
        gstapp-1.0
        libgstapp-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstapp library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstbase_LIBRARY
    NAMES
        gstbase-1.0
        libgstbase-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstbase library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GLIB_gstcdda_LIBRARY
    NAMES
        gstcdda-1.0
        libgstcdda-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstcdda library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstcontroller_LIBRARY
    NAMES
        gstcontroller-1.0
        libgstcontroller-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstcontroller library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstdataprotocol_LIBRARY
    NAMES
        gstdataprotocol-1.0
        libgstdataprotocol-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstdataprotocol library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstnet_LIBRARY
    NAMES
        gstnet-1.0
        libgstnet-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstnet library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstnetbuffer_LIBRARY
    NAMES
        gstnetbuffer-1.0
        libgstnetbuffer-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstnetbuffer library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstpbutils_LIBRARY
    NAMES
        gstpbutils-1.0
        libgstpbutils-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstpbutils library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstreamer_LIBRARY
    NAMES
        gstreamer-1.0
        libgstreamer-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstreamer library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstriff_LIBRARY
    NAMES
        gstriff-1.0
        libgstriff-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstriff library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstrtp_LIBRARY
    NAMES
        gstrtp-1.0
        libgstrtp-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstrtp library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstrtsp_LIBRARY
    NAMES
        gstrtsp-1.0
        libgstrtsp-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstrtsp library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstrtspserver_LIBRARY
    NAMES
        gstrtspserver-1.0
        libgstrtspserver-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstrtsp library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstsdp_LIBRARY
    NAMES
        gstsdp-1.0
        libgstsdp-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstsdp library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gsttag_LIBRARY
    NAMES
        gsttag-1.0
        libgsttag-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gsttag library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
)

find_library(GSTREAMER_gstvideo_LIBRARY
    NAMES
        gstvideo-1.0 libgstvideo-1.0
    PATHS ${gstreamer_library_search_paths}
    ENV LIB
    DOC "gstvideo library to link with"
    NO_SYSTEM_ENVIRONMENT_PATH
    )

find_library(GOBJECT_LIBRARY NAMES libobject-2.0 gobject-2.0
    PATHS ${GSTREAMER_DIR}/lib
    ENV LIB
    DOC "Glib library"
    NO_SYSTEM_ENVIRONMENT_PATH
)

SET(GSTREAMER_FOUND FALSE)
set(GStreamer_FOUND FALSE)
IF (GSTREAMER_gst_INCLUDE_DIR AND GSTREAMER_gstreamer_LIBRARY AND GSTREAMER_gstapp_LIBRARY)

  SET(GSTREAMER_INCLUDE_DIR ${GSTREAMER_gst_INCLUDE_DIR} ${GSTREAMER_gstconfig_INCLUDE_DIR})
  list(REMOVE_DUPLICATES GSTREAMER_INCLUDE_DIR)
  SET(GSTREAMER_LIBRARIES ${GSTREAMER_gstreamer_LIBRARY} GSTREAMER_gstapp_LIBRARY)
  list(REMOVE_DUPLICATES GSTREAMER_LIBRARIES)
  SET(GSTREAMER_FOUND TRUE)
  set(GStreamer_FOUND TRUE)

ENDIF (GSTREAMER_gst_INCLUDE_DIR AND GSTREAMER_gstreamer_LIBRARY AND GSTREAMER_gstapp_LIBRARY)
