set(plugin_export_template_path "${CMAKE_CURRENT_LIST_DIR}/PluginExport.hpp.in" CACHE INTERNAL "")
macro(aquila_declare_plugin tgt)
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

    set_target_properties(${tgt} PROPERTIES FOLDER Plugins)
    set_target_properties(${tgt}
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Plugins
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Plugins
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Plugins
    )
    set_target_properties(${tgt} PROPERTIES LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/Plugins)


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

    RCC_TARGET_CONFIG(${tgt})

    set(${tgt}_PLUGIN_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src/" CACHE PATH "" FORCE)
    target_include_directories(${tgt}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
    )

    ADD_DEFINITIONS(-DPLUGIN_NAME=${tgt})
    set(PLUGIN_NAME "${tgt}")
    CONFIGURE_FILE(${plugin_export_template_path} "${CMAKE_CURRENT_LIST_DIR}/src/${tgt}Export.hpp" @ONLY)


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
    set(link_file_path "${CMAKE_CURRENT_LIST_DIR}/${tgt}/src/Aquila/rcc/external_includes/${tgt}_link_libs.hpp")

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

    FILE(WRITE "${CMAKE_CURRENT_LIST_DIR}/${tgt}/src/.gitignore" "Aquila\n${tgt}Export.hpp\nprecompiled.hpp")

    INSTALL(TARGETS ${tgt}
            LIBRARY DESTINATION bin/Plugins
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
