
function(aquila_install_dependent_lib FILENAME)

    if(EXISTS ${FILENAME})
        if(IS_SYMLINK ${FILENAME})
            get_filename_component(path ${FILENAME} REALPATH)
            INSTALL(FILES ${path} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
            INSTALL(FILES ${FILENAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        else()
            INSTALL(FILES ${FILENAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        endif()
    endif()

endfunction(aquila_install_dependent_lib)
