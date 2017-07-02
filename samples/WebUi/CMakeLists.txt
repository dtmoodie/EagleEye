
find_package(Wt QUIET)
if(Wt_FOUND AND TARGET metaobject_wtgui)
  find_package(CUDA REQUIRED)
  include_directories(${Wt_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  file(GLOB_RECURSE src "*.cpp" "*.hpp")
  add_executable(WebUi ${src})
  target_link_libraries(WebUi
    aquila_core
    metaobject_core
    metaobject_metaparams
    metaobject_wtgui
  )

  IF(WIN32)
  TARGET_COMPILE_OPTIONS(WebUi PUBLIC "/bigobj")
  set(PROJECT_BIN_DIRS_DEBUG "")
  set(PROJECT_BIN_DIRS_RELEASE "")
  set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
  MACRO(APPEND_BIN_DIR ARG1 ARG2)
    IF(EXISTS "${${ARG1}}" AND IS_DIRECTORY "${${ARG1}}")
        LIST(APPEND ${ARG2} ${${ARG1}})
    ELSE()
        if(RCC_VERBOSE_CONFIG)
          MESSAGE("${ARG1} : ${${ARG1}} is not a valid directory")
        endif()
    ENDIF()
  ENDMACRO(APPEND_BIN_DIR)

  FOREACH(dir ${BIN_DIRS})
    APPEND_BIN_DIR(${dir}_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
    APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
    APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)
  ENDFOREACH(dir ${BIN_DIRS})
  CONFIGURE_FILE("../../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/samples/WebUi/WebUi.vcxproj.user @ONLY)
endif(WIN32)
else(Wt_FOUND AND TARGET metaobject_wtgui)
  if(RCC_VERBOSE_CONFIG)
    message(STATUS "Wt not found, not building web interface")
  endif(RCC_VERBOSE_CONFIG)
endif(Wt_FOUND AND TARGET metaobject_wtgui)
