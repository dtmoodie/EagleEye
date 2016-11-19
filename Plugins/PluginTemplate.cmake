macro(RCC_HANDLE_LIB TARGET)
    if(RCC_VERBOSE_CONFIG)
      message(STATUS "===================================================================
               RCC config information for ${TARGET}")
    endif(RCC_VERBOSE_CONFIG)
    foreach(lib ${ARGN})
    endforeach(lib ${ARGN})
endmacro(RCC_HANDLE_LIB target lib)

get_target_property(target_include_dirs_ ${PROJECT_NAME} INCLUDE_DIRECTORIES)
get_target_property(target_link_libs_    ${PROJECT_NAME} LINK_LIBRARIES)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Plugins)
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Plugins)

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
        "include_dirs:\n${target_include_dirs_}\n"
        "lib_dirs_debug:\n${link_dirs_debug}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug\n"
        "lib_dirs_release:\n${link_dirs_release}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo\n"
        "compile_options:\n/DPROJECT_BUILD_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\" ${WIN_DEFS} /DPLUGIN_NAME=${PROJECT_NAME} /FI\"EagleLib/Detail/PluginExport.hpp\""
    )

    FILE(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo/${PROJECT_NAME}_config.txt" 
        "project_id:\n${PROJECT_ID}\n"
        "include_dirs:\n${target_include_dirs_}\n"
        "lib_dirs_debug:\n${link_dirs_debug}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug\n"
        "lib_dirs_release:\n${link_dirs_release}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo\n"
        "compile_options:\n/DPROJECT_BUILD_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\" ${WIN_DEFS} /DPLUGIN_NAME=${PROJECT_NAME} /FI\"EagleLib/Detail/PluginExport.hpp\""
    )
	set(outfile_ "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug/${PROJECT_NAME}_config.txt")
else(WIN32)
    string(REGEX REPLACE ";" "\n" WIN_DEFS "${DEFS}")

    FILE(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}_config.txt"
        "project_id:\n${PROJECT_ID}\n"
        "\ninclude_dirs:\n${target_include_dirs_}\n"
        "\nlib_dirs_debug:\n${link_dirs_debug}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug\n"
        "\nlib_dirs_release:\n${link_dirs_release}\n${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/RelWithDebInfo\n"
        "\ncompile_options:\n-DPROJECT_BUILD_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\"\n${WIN_DEFS}\n-DPLUGIN_NAME=${PROJECT_NAME}\n-include \"EagleLib/Detail/PluginExport.hpp\""
    )
	set(outfile_ "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}_config.txt")

endif(WIN32)

ADD_DEFINITIONS(-DPLUGIN_NAME=${PROJECT_NAME})
set(PLUGIN_NAME "${PROJECT_NAME}")
CONFIGURE_FILE("../PluginExport.hpp.in" "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/src/${PROJECT_NAME}Export.hpp" @ONLY)
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FI\"EagleLib/Detail/PluginExport.hpp\"")
else(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include \"EagleLib/Detail/PluginExport.hpp\"")
endif(WIN32)

LINK_DIRECTORIES(${LINK_DIRS_DEBUG})
LINK_DIRECTORIES(${LINK_DIRS_RELEASE})
LINK_DIRECTORIES(${LINK_DIRS})

INSTALL(TARGETS ${PROJECT_NAME}
	LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
       
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
 ")
ENDIF(RCC_VERBOSE_CONFIG)
