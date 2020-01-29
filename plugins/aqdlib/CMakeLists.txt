project(dlib)

find_package(dlib QUIET)
if(NOT dlib_FOUND)
    set(DLIB_USE_CUDA ON)
    set(DLIB_IN_PROJECT_BUILD OFF)
    add_subdirectory(dependencies/dlib)
    set_plugin_status(aqdlib "Dlib install not found, building as part of plugin")
    set(DLIB_SHARED_LIB dlib)
    set(DLIB_STATIC_LIB dlib)
else()

    INCLUDE_DIRECTORIES(${dlib_INCLUDE_DIRS})
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
    set_plugin_status(aqdlib "Dlib install found. ${dlib_DIR} include: ${dlib_INCLUDE_DIRS} lib: ${DLIB_SHARED_LIB}")
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
    ${CUDA_CUBLAS_LIBRARIES}
    metaobject_core
    metaobject_object
    metaobject_params
    ${DLIB_SHARED_LIB}
    ${dlib_needed_libraries}
    lapack
    aqcore
)
aq_install_python_test(tests/face_recognition.py)
aq_install_python_test(scripts/door_face.py)
file(GLOB_RECURSE MODELS "share/*")
add_custom_target(aqdlib_models ${MODELS})
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
