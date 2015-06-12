cmake_minimum_required(VERSION 2.8)

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


project(Plugins)
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
FOREACH(subdir ${SUBDIRS})
    ADD_SUBDIRECTORY(${subdir})
ENDFOREACH()

#add_subdirectory(fastms)
#add_subdirectory(PCL_bridge)
#add_subdirectory(Plotting)
#add_subdirectory(Dev)
#add_subdirectory(Caffe)
#add_subdirectory(Segmentation)

