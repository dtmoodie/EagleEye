# Main variables:
# OPENNI2_LIBRARY and OPENNI2_INCLUDES to link OpenCV modules with OpenNI2
# HAVE_OPENNI2 for conditional compilation OpenCV with/without OpenNI2

if(NOT "${OPENNI2_LIB_DIR}" STREQUAL "${OPENNI2_LIB_DIR_INTERNAL}")
    unset(OPENNI2_LIBRARY CACHE)
    unset(OPENNI2_LIB_DIR CACHE)
endif()

if(NOT "${OPENNI2_INCLUDE_DIR}" STREQUAL "${OPENNI2_INCLUDE_DIR_INTERNAL}")
    #unset(OPENNI2_INCLUDES CACHE)
    #unset(OPENNI2_INCLUDE_DIR CACHE)
endif()

if(WIN32)
    if(NOT (MSVC64 OR MINGW64))
        find_file(OPENNI2_INCLUDES "OpenNI.h" PATHS "$ENV{OPEN_NI_INSTALL_PATH}Include" DOC "OpenNI2 c++ interface header")
        find_library(OPENNI2_LIBRARY "OpenNI2" PATHS $ENV{OPENNI2_LIB} DOC "OpenNI2 library")
    else()
        find_file(OPENNI2_INCLUDES "OpenNI.h" PATHS $ENV{OPENNI2_INCLUDE64} "$ENV{OPEN_NI_INSTALL_PATH64}Include" DOC "OpenNI2 c++ interface header")
        find_library(OPENNI2_LIBRARY "OpenNI2" PATHS $ENV{OPENNI2_LIB64} DOC "OpenNI2 library")
    endif()
elseif(UNIX OR APPLE)
    find_file(OPENNI2_INCLUDES "OpenNI.h" PATHS "/usr/include/ni2" "/usr/include/openni2" $ENV{OPENNI2_INCLUDE} DOC "OpenNI2 c++ interface header")
    find_library(OPENNI2_LIBRARY "OpenNI2" PATHS "/usr/lib" $ENV{OPENNI2_REDIST} DOC "OpenNI2 library")
endif()

if(OPENNI2_LIBRARY AND OPENNI2_INCLUDES)
    set(HAVE_OPENNI2 TRUE)
endif() #if(OPENNI2_LIBRARY AND OPENNI2_INCLUDES)

get_filename_component(OPENNI2_LIB_DIR "${OPENNI2_LIBRARY}" PATH)
get_filename_component(OPENNI2_INCLUDE_DIR ${OPENNI2_INCLUDES} PATH)

if(HAVE_OPENNI2)
  set(OPENNI2_LIB_DIR "${OPENNI2_LIB_DIR}" CACHE PATH "Path to OpenNI2 libraries" FORCE)
  set(OPENNI2_INCLUDE_DIR "${OPENNI2_INCLUDE_DIR}" CACHE PATH "Path to OpenNI2 headers" FORCE)
endif()

if(OPENNI2_LIBRARY)
    set(OPENNI2_LIB_DIR_INTERNAL "${OPENNI2_LIB_DIR}" CACHE INTERNAL "This is the value of the last time OPENNI_LIB_DIR was set successfully." FORCE)
else()
    if(OpenNI2_FIND_REQUIRED)
        message( FATAL " OpenNI2 library directory (set by OPENNI2_LIB_DIR variable) is not found or does not have OpenNI2 libraries." )
    else()
        if(NOT OpenNI2_FIND_QUIETLY)
            message( WARNING " OpenNI2 library directory (set by OPENNI2_LIB_DIR variable) is not found or does not have OpenNI2 libraries." )
        endif()
    endif()
endif()

if(OPENNI2_INCLUDES)
    set(OPENNI2_INCLUDE_DIR_INTERNAL "${OPENNI2_INCLUDE_DIR}" CACHE INTERNAL "This is the value of the last time OPENNI2_INCLUDE_DIR was set successfully." FORCE)
else()
    if(NOT OpenNI2_FIND_QUIETLY)
        message( WARNING, " OpenNI2 include directory (set by OPENNI2_INCLUDE_DIR variable) is not found or does not have OpenNI2 include files." )
    endif()
endif()

mark_as_advanced(FORCE OPENNI2_LIBRARY)
mark_as_advanced(FORCE OPENNI2_INCLUDES)


