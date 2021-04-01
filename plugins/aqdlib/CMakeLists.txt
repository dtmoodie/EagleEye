project(dlib)

find_package(dlib QUIET)
find_package(CUDNN QUIET)
if(CUDNN_FOUND)
    if(NOT dlib_FOUND)
        set(DLIB_USE_CUDA ON)
        set(DLIB_IN_PROJECT_BUILD OFF)
        add_subdirectory(dependencies/dlib)
        set_plugin_status(aqdlib "Dlib install not found, building as part of plugin.  cudnn version found: ${CUDNN_VERSION}")
        set(DLIB_SHARED_LIB dlib)
        set(DLIB_STATIC_LIB dlib)
    else()
        print_target_properties(dlib::dlib_shared)
        if(TARGET dlib::dlib_shared)
            set(DLIB_SHARED_LIB dlib::dlib_shared)
        else()
            find_library(DLIB_SHARED_LIB dlib libdlib.so
                HINTS ${dlib_DIR}/lib
                    ${dlib_DIR}/../../
            )
        endif()
        if(TARGET dlib::dlib)
            set(DLIB_STATIC_LIB dlib::dlib)
        else()
            find_library(DLIB_STATIC_LIB libdlib.a
                HINTS ${dlib_DIR}/lib
            )
        endif()

        find_package(OpenBLAS QUIET)
        if(NOT OpenBLAS_FOUND)
            set_plugin_unavailable(aqdlib)
            set_plugin_status(aqdlib "Dlib install found. But could not find openblas")
            return()
        endif()
        set(DLIB_SHARED_LIB "${DLIB_SHARED_LIB}")
        set_plugin_status(aqdlib "Dlib install found. ${dlib_DIR} lib: ${DLIB_SHARED_LIB}.  cudnn version found: ${CUDNN_VERSION}")
        set(dlib_needed_libraries ${dlib_needed_libraries};${OpenBLAS_LIB})
    endif()

    file(GLOB_RECURSE knl "src/*.cu")
    file(GLOB_RECURSE src "src/*.cpp")
    file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")

    add_library(aqdlib SHARED ${src} ${hdr} ${knl})

    target_link_libraries(aqdlib
        aquila_utilities
        aquila_serialization
        aquila_types
        ${OpenCV_LIBS}
        ${CUDA_cublas_LIBRARY}
        metaobject_core
        metaobject_object
        metaobject_params
        ${DLIB_SHARED_LIB}
        ${dlib_needed_libraries}
        lapack
        aqcore
    )
    aq_install_python_test(tests/face_recognition.py)
    aq_install_python_test(tests/face_recognition_from_file.py)
    aq_install_python_test(scripts/door_face.py)
    #if(TARGET aquila_python)
        get_target_property(out_dir aquila_python ARCHIVE_OUTPUT_DIRECTORY)
        add_custom_command(TARGET aqdlib POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_CURRENT_LIST_DIR}/data/images/
                        $<TARGET_FILE_DIR:aquila_python>/images
        )
    #endif()

    configure_file(${CMAKE_CURRENT_LIST_DIR}/data/identities.db ${out_dir}/identities.db COPYONLY)
    file(GLOB_RECURSE MODELS "share/*")
    add_custom_target(aqdlib_models ${MODELS})
    if(NOT dlib_FOUND)
        export(TARGETS dlib FILE "${PROJECT_BINARY_DIR}/dlibTargets.cmake")
    endif()
    aquila_declare_plugin(aqdlib)
    set_plugin_available(aqdlib)
    aquila_install_dependent_lib(${DLIB_SHARED_LIB})

    install(
        DIRECTORY share/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/aqdlib
    )

    install(DIRECTORY scripts/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/aqdlib
    )
else(CUDNN_FOUND)
    set_plugin_unavailable(aqdlib)
    set_plugin_status(aqdlib "Unable to find cudnn, thus disabling dlib plugin")
endif(CUDNN_FOUND)