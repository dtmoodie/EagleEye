set(plugin_export_template_path "${CMAKE_CURRENT_LIST_DIR}/PluginExport.hpp.in" CACHE INTERNAL "")
macro(aquila_declare_plugin tgt)
    set(options SVN)
    cmake_parse_arguments(aquila_declare_plugin "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    macro(RCC_HANDLE_LIB TARGET)
      if(RCC_VERBOSE_CONFIG)
        message(STATUS "===================================================================\n"
                       "  RCC config information for ${TARGET}")
      endif(RCC_VERBOSE_CONFIG)
      foreach(lib ${ARGN})
      endforeach(lib ${ARGN})
    endmacro(RCC_HANDLE_LIB target lib)

    get_target_property(target_include_dirs_ ${tgt} INCLUDE_DIRECTORIES)
    get_target_property(target_link_libs_    ${tgt} LINK_LIBRARIES)

    set_target_properties(${tgt} PROPERTIES FOLDER plugins)
    set_target_properties(${tgt}
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
    )
    set_target_properties(${tgt} PROPERTIES LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/plugins)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/src)
        target_include_directories(${tgt}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
    )
        target_include_directories(${tgt}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/plugins/${tgt}/>
    )

    RCC_TARGET_CONFIG(${tgt} plugin_libraries_debug plugin_libraries_release)

    ocv_add_precompiled_header_to_target(${tgt} src/precompiled.hpp)
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${tgt}_config.txt")
      FILE(READ "${CMAKE_CURRENT_BINARY_DIR}/${tgt}_config.txt" temp)
    endif()

    SET(PROJECT_ID)
    IF(temp)
      STRING(FIND "${temp}" "\n" len)
      STRING(SUBSTRING "${temp}" 0 ${len} temp)
      SET(PROJECT_ID ${temp})
      if(RCC_VERBOSE_CONFIG)
        message("Project ID for ${tgt}: ${PROJECT_ID}")
      endif()
    ELSE(temp)
      SET(PROJECT_ID "1")
    ENDIF(temp)

    set(LINK_LIBS_RELEASE ${plugin_libraries_release})
    set(LINK_LIBS_DEBUG ${plugin_libraries_debug})

    set(${tgt}_PLUGIN_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src/" CACHE PATH "" FORCE)

    set(PLUGIN_NAME ${tgt})
    string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M")
    execute_process(
          COMMAND ${GITCOMMAND} rev-parse --abbrev-ref HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../Aquila
          OUTPUT_VARIABLE AQUILA_GIT_BRANCH
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        execute_process(
          COMMAND ${GITCOMMAND} log -1 --format=%H
          WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../Aquila
          OUTPUT_VARIABLE AQUILA_GIT_COMMIT_HASH
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        execute_process(
          COMMAND ${GITCOMMAND} rev-parse --abbrev-ref HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../Aquila/dependencies/MetaObject
          OUTPUT_VARIABLE MO_GIT_BRANCH
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        execute_process(
          COMMAND ${GITCOMMAND} log -1 --format=%H
          WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../Aquila/dependencies/MetaObject
          OUTPUT_VARIABLE MO_GIT_COMMIT_HASH
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        execute_process(
                COMMAND ${GITCOMMAND} status
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                RESULT_VARIABLE IS_GIT_REPO
                OUTPUT_VARIABLE _IGNORE_
                ERROR_QUIET
        )

        if(NOT ${IS_GIT_REPO} AND NOT ${aquila_declare_plugin_SVN}) # return code of 0 if success
        set(REPO_TYPE "git")
                execute_process(
                  COMMAND ${GITCOMMAND} rev-parse --abbrev-ref HEAD
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE GIT_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET
                )
                execute_process(
                  COMMAND ${GITCOMMAND} log -1 --format=%H
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE GIT_COMMIT_HASH
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET
                )

                execute_process(
                  COMMAND ${GITCOMMAND} config user.name
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE GIT_USERNAME
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET
                )
                execute_process(
                  COMMAND ${GITCOMMAND} config user.email
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE GIT_EMAIL
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET
                )
        ELSE(NOT ${IS_GIT_REPO} AND NOT ${aquila_declare_plugin_SVN})
                # check if it's an SVN repoo
                IF(SVNCOMMAND)
                        execute_process(
                                COMMAND ${SVNCOMMAND} status
                                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                                OUTPUT_VARIABLE _IGNORE_
                                RESULT_VARIABLE IS_SVN_REPO
                                ERROR_QUIET
                        )
                        IF(NOT ${IS_SVN_REPO}) # return code is 0 if success
                set(REPO_TYPE "svn")
                                execute_process(
                                        COMMAND ${SVNCOMMAND} info --show-item revision
                                        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                                        OUTPUT_VARIABLE GIT_COMMIT_HASH
                                        ERROR_QUIET
                                )
                string(REGEX REPLACE "\n" "" GIT_COMMIT_HASH "${GIT_COMMIT_HASH}")
                execute_process(
                  COMMAND ${SVNCOMMAND} info --show-item relative-url
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE GIT_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET
                )
                if(WIN32)
                    set(GIT_USERNAME "$ENV{USERNAME}")
                else()
                    set(GIT_USERNAME "$ENV{USER}")
                endif()
                        endif(NOT ${IS_SVN_REPO})
                endif()
    ENDIF(NOT ${IS_GIT_REPO} AND NOT ${aquila_declare_plugin_SVN})

    CONFIGURE_FILE(${plugin_export_template_path} "${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_export.hpp" @ONLY)

    CONFIGURE_FILE("../plugin_config.cpp.in" "${CMAKE_BINARY_DIR}/plugins/${tgt}/plugin_config.cpp" @ONLY)

    set_property(TARGET ${tgt} APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/plugins/${tgt}/plugin_config.cpp")
    set_property(TARGET ${tgt} APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_export.hpp")

    LINK_DIRECTORIES(${LINK_DIRS_DEBUG})
    LINK_DIRECTORIES(${LINK_DIRS_RELEASE})
    LINK_DIRECTORIES(${LINK_DIRS})

    # ============= Write out a file containing external include info

    set(external_include_file "#pragma once\n\n#include \"RuntimeObjectSystem/RuntimeLinkLibrary.h\"\n\n")
    # wndows link libs
    if(LINK_LIBS_RELEASE)
      LIST(REMOVE_DUPLICATES LINK_LIBS_RELEASE)
          list(SORT LINK_LIBS_RELEASE)
    endif()
    if(LINK_LIBS_DEBUG)
      LIST(REMOVE_DUPLICATES LINK_LIBS_DEBUG)
          list(SORT LINK_LIBS_DEBUG)
    endif()
    set(external_include_file "${external_include_file}\n#if defined(NDEBUG) && !defined(_DEBUG)\n\n")
        if(WIN32)
                set(prefix "")
                set(postfix ".lib")
        else(WIN32)
                set(prefix "-l")
                set(postfix "")
        endif(WIN32)
    foreach(lib ${LINK_LIBS_RELEASE})
        string(LENGTH ${lib} len)
        if(len GREATER 3)
            string(SUBSTRING "${lib}" 0 3 sub)
            if(${sub} STREQUAL lib)
              string(SUBSTRING "${lib}" 3 -1 lib)
              set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            else()
              set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            endif()
        else()
            set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
        endif()
    endforeach()

    set(external_include_file "${external_include_file}\n  #else\n\n")

    foreach(lib ${LINK_LIBS_DEBUG})
        string(LENGTH ${lib} len)
        if(len GREATER 3)
            string(SUBSTRING "${lib}" 0 3 sub)
            if(${sub} STREQUAL lib)
                string(SUBSTRING "${lib}" 3 -1 lib)
                set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            else()
                set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            endif()
        else()
            set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
        endif()
    endforeach()

    set(external_include_file "${external_include_file}\n  #endif // NDEBUG\n")

    set(external_include_file "${external_include_file}\n")
    set(link_file_path "${CMAKE_BINARY_DIR}/plugins/${tgt}/Aquila/rcc/external_includes/${tgt}_link_libs.hpp")

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

    INSTALL(TARGETS ${tgt}
            LIBRARY DESTINATION bin/plugins
            RUNTIME DESTINATION bin
    )

    export(TARGETS ${tgt} FILE "${PROJECT_BINARY_DIR}/${tgt}Targets.cmake")
    export(PACKAGE ${tgt})

    install(TARGETS ${tgt}
        DESTINATION lib
        EXPORT ${tgt}Targets
    )

    INSTALL(DIRECTORY src/ DESTINATION include/${tgt} FILES_MATCHING PATTERN "*.hpp")
    INSTALL(DIRECTORY src/ DESTINATION include/${tgt} FILES_MATCHING PATTERN "*.h")
    install(EXPORT ${tgt}Targets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Aquila" COMPONENT dev)

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

endmacro(aquila_declare_plugin)
