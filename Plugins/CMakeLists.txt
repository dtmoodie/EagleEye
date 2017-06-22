cmake_minimum_required(VERSION 2.8)
include(PluginTemplate.cmake)
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

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined")
else()
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/Plugins")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/Plugins")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/Plugins")

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/Plugins")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/Plugins")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/Plugins")

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/Plugins")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/Plugins")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/Plugins")
endif()

project(Plugins)
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "------------------- Plugins -------------------------")
FOREACH(subdir ${SUBDIRS})
  SET(Plugin_${subdir} true CACHE BOOL "")
  IF(Plugin_${subdir})
    set(PLUGIN_NAME ${subdir})
    string(TIMESTAMP BUILD_DATE "%Y-%m-%d:%H:%M")
    
    execute_process(
      COMMAND git rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../Aquila
      OUTPUT_VARIABLE AQUILA_GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND git log -1 --format=%H
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../Aquila
      OUTPUT_VARIABLE AQUILA_GIT_COMMIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    execute_process(
      COMMAND git rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../Aquila/dependencies/MetaObject
      OUTPUT_VARIABLE MO_GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND git log -1 --format=%H
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../Aquila/dependencies/MetaObject
      OUTPUT_VARIABLE MO_GIT_COMMIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    
    execute_process(
      COMMAND git rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${subdir}
      OUTPUT_VARIABLE GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND git log -1 --format=%H
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${subdir}
      OUTPUT_VARIABLE GIT_COMMIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
      COMMAND git config user.name
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${subdir}
      OUTPUT_VARIABLE GIT_USERNAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND git config user.email
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${subdir}
      OUTPUT_VARIABLE GIT_EMAIL
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    CONFIGURE_FILE("plugin_config.cpp.in" "${CMAKE_CURRENT_LIST_DIR}/${subdir}/src/plugin_config.cpp" @ONLY)
    ADD_SUBDIRECTORY(${subdir})
    if(Plugin_${subdir}_available)
      message(STATUS "Plugin ${subdir}: ${Plugin_${subdir}_status}")
    endif()
  ENDIF()
ENDFOREACH()
message(STATUS "------------------- End Plugins ----------------------")

