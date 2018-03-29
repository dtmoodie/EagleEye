project(dlib)

find_package(dlib QUIET)
set(DLIB_USE_CUDA ON)
set(DLIB_IN_PROJECT_BUILD OFF)
add_subdirectory(dependencies/dlib)
set_plugin_status(aqdlib "Dlib install not found, building as part of plugin")

file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")

add_library(aqdlib SHARED ${src} ${hdr} ${knl})

target_link_libraries(aqdlib
    aquila_utilities
    aquila_serialization
    aquila_types
    aquila_metatypes
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
    metaobject_core
    metaobject_object
    metaobject_params
    dlib
    )
aq_install_python_test(tests/face_recognition.py)
aquila_declare_plugin(aqdlib)
set_plugin_available(aqdlib)
