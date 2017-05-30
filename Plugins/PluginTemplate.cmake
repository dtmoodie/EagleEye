macro(RCC_HANDLE_LIB TARGET)
  if(RCC_VERBOSE_CONFIG)
    message(STATUS "===================================================================\n"
                   "  RCC config information for ${TARGET}")
  endif(RCC_VERBOSE_CONFIG)
  foreach(lib ${ARGN})
  endforeach(lib ${ARGN})
endmacro(RCC_HANDLE_LIB target lib)

get_target_property(target_include_dirs_ ${PROJECT_NAME} INCLUDE_DIRECTORIES)
get_target_property(target_link_libs_    ${PROJECT_NAME} LINK_LIBRARIES)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Plugins)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Plugins
        CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Plugins
        CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Plugins
)
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/Plugins)


ocv_add_precompiled_header_to_target(${PROJECT_NAME} src/precompiled.hpp)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_config.txt")
  FILE(READ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_config.txt" temp)
endif()

SET(PROJECT_ID)
IF(temp)
  STRING(FIND "${temp}" "\n" len)
  STRING(SUBSTRING "${temp}" 0 ${len} temp)
  SET(PROJECT_ID ${temp})
  if(RCC_VERBOSE_CONFIG)
    message("Project ID for ${PROJECT_NAME}: ${PROJECT_ID}")
  endif()
ELSE(temp)
  SET(PROJECT_ID "1")
ENDIF(temp)

foreach( lib ${target_link_libs_})
    if(EXISTS ${lib})
        GET_FILENAME_COMPONENT(dir ${lib} DIRECTORY)
        if(dir)
            if(RCC_VERBOSE_CONFIG)
                message(STATUS "Library ${lib} link directory: ${dir}")
            endif()
            list(APPEND LINK_DIRS_RELEASE ${dir})
            list(APPEND LINK_DIRS_DEBUG ${dir})
        endif()
    endif()
endforeach()


LIST(REMOVE_DUPLICATES target_include_dirs_)
LIST(REMOVE_DUPLICATES LINK_DIRS_RELEASE)
LIST(REMOVE_DUPLICATES LINK_DIRS_DEBUG)
set(outfile_ "")

string(REGEX REPLACE ";" "\n" target_include_dirs_ "${target_include_dirs_}")
string(REGEX REPLACE ";" "\n" link_dirs_release "${LINK_DIRS_RELEASE}")
string(REGEX REPLACE ";" "\n" link_dirs_debug "${LINK_DIRS_DEBUG}")

if(WIN32)
  string(REGEX REPLACE "-D" "/D" WIN_DEFS "${DEFS}")
  string(REGEX REPLACE ";" "\n" WIN_DEFS "${WIN_DEFS}")
  FILE(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug/${PROJECT_NAME}_config.txt"
        "project_id:\n${PROJECT_ID}\n"
        "include_dirs:\n${target_include_dirs_}\n${CMAKE_CURRENT_LIST_DIR}/src\n"
        "lib_dirs_debug:\n${link_dirs_debug}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug\n"
        "lib_dirs_release:\n${link_dirs_release}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo\n"
        "compile_options:\n/DPROJECT_BUILD_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\" ${WIN_DEFS} /DPLUGIN_NAME=${PROJECT_NAME}"
  )

  FILE(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo/${PROJECT_NAME}_config.txt"
        "project_id:\n${PROJECT_ID}\n"
        "include_dirs:\n${target_include_dirs_}\n${CMAKE_CURRENT_LIST_DIR}/src\n"
        "lib_dirs_debug:\n${link_dirs_debug}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug\n"
        "lib_dirs_release:\n${link_dirs_release}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo\n"
        "compile_options:\n/DPROJECT_BUILD_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\" ${WIN_DEFS} /DPLUGIN_NAME=${PROJECT_NAME}"
  )
  set(outfile_ "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug/${PROJECT_NAME}_config.txt")
else(WIN32)
  string(REGEX REPLACE ";" "\n" WIN_DEFS "${DEFS}")

  FILE(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}_config.txt"
       "project_id:\n"
       "${PROJECT_ID}\n"
       "\ninclude_dirs:\n"
       "${target_include_dirs_}\n"
       "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/src\n"
       "\nlib_dirs_debug:\n"
       "${link_dirs_debug}\n"
       "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug\n"
       "\n"
       "lib_dirs_release:\n"
       "${link_dirs_release}\n"
       "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo\n"
       "\n"
       "compile_options:\n"
       "-DPROJECT_BUILD_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\"\n"
       "${WIN_DEFS}\n"
       "-DPLUGIN_NAME=${PROJECT_NAME}\n"
    )
  set(outfile_ "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}_config.txt")
endif(WIN32)

set(${PROJECT_NAME}_PLUGIN_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/src/" CACHE PATH "" FORCE)

ADD_DEFINITIONS(-DPLUGIN_NAME=${PROJECT_NAME})
set(PLUGIN_NAME "${PROJECT_NAME}")
CONFIGURE_FILE("${CMAKE_CURRENT_LIST_DIR}/PluginExport.hpp.in" "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/src/${PROJECT_NAME}Export.hpp" @ONLY)


LINK_DIRECTORIES(${LINK_DIRS_DEBUG})
LINK_DIRECTORIES(${LINK_DIRS_RELEASE})
LINK_DIRECTORIES(${LINK_DIRS})

# ============= Write out a file containing external include info

set(external_include_file "#pragma once\n\n#include \"RuntimeObjectSystem/RuntimeLinkLibrary.h\"\n\n#ifdef _MSC_VER\n")
# wndows link libs
if(LINK_LIBS_RELEASE)
  LIST(REMOVE_DUPLICATES LINK_LIBS_RELEASE)
endif()
if(LINK_LIBS_DEBUG)
  LIST(REMOVE_DUPLICATES LINK_LIBS_DEBUG)
endif()
set(external_include_file "${external_include_file}\n#else\n\n  #ifdef NDEBUG\n")

foreach(lib ${LINK_LIBS_RELEASE})
    string(LENGTH ${lib} len)
    if(len GREATER 3)
        string(SUBSTRING "${lib}" 0 3 sub)
        if(sub STREQUAL lib)
            MESSAGE(${lib})
            MESSAGE(${sub})
          string(SUBSTRING "${lib}" 3 -1 lib)
          set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"-l${lib}\")\n")
        else()
          set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"-l${lib}\")\n")
        endif()
    else()
        set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"-l${lib}\")\n")
    endif()

endforeach()

set(external_include_file "${external_include_file}\n  #else\n")

foreach(lib ${LINK_LIBS_DEBUG})
    set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"-l${lib}\")\n")
endforeach()
set(external_include_file "${external_include_file}\n  #endif // NDEBUG\n")

set(external_include_file "${external_include_file}\n#endif // _MSC_VER")
set(link_file_path "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/src/Aquila/rcc/external_includes/${PROJECT_NAME}_link_libs.hpp")

if(EXISTS ${link_file_path})
    FILE(READ ${link_file_path} read_file)
    if(NOT ${read_file} STREQUAL ${external_include_file})
        FILE(WRITE ${link_file_path} "${external_include_file}")
        if(RCC_VERBOSE_CONFIG)
            message(STATUS "Updating ${link_file_path}")
        endif()
    endif()
else()
  FILE(WRITE ${link_file_path} "${external_include_file}")
endif()

FILE(WRITE "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/src/.gitignore" "Aquila\n${PROJECT_NAME}Export.hpp\nprecompiled.hpp")

INSTALL(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION bin/Plugins
        RUNTIME DESTINATION bin
)

export(TARGETS ${PROJECT_NAME} FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export(PACKAGE ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
    DESTINATION lib
    EXPORT ${PROJECT_NAME}Targets
)

INSTALL(DIRECTORY src/ DESTINATION include/${PROJECT_NAME} FILES_MATCHING PATTERN "*.hpp")
INSTALL(DIRECTORY src/ DESTINATION include/${PROJECT_NAME} FILES_MATCHING PATTERN "*.h")
install(EXPORT ${PROJECT_NAME}Targets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Aquila" COMPONENT dev)

IF(RCC_VERBOSE_CONFIG)
  string(REGEX REPLACE ";" "\n    " include_dirs_ "${INCLUDE_DIRS}")
  string(REGEX REPLACE ";" "\n    " link_dirs_release_ "${LINK_DIRS_RELEASE}")
  string(REGEX REPLACE ";" "\n    " link_dirs_debug_ "${LINK_DIRS_DEBUG}")
  MESSAGE(STATUS
  "  ${outfile_}
  Include Dirs:
    ${include_dirs_}
  Link Dirs Debug:
    ${link_dirs_debug_}
  Link Dirs Release:
    ${link_dirs_release_}
  Link libs Release:
    ${LINK_LIBS_RELEASE}
  Link libs Debug:
    ${LINK_LIBS_DEBUG}
 ")
ENDIF(RCC_VERBOSE_CONFIG)
