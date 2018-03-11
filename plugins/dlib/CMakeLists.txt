project(dlib)

add_subdirectory(dependencies/dlib)

file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")

cuda_add_library(aqdlib SHARED ${src} ${hdr} ${knl})

target_link_libraries(aqdlib aquila_utilities
    aquila_serialization
    aquila_types
    aquila_metatypes
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
    metaobject_core
    metaobject_object
    metaobject_params
    dlib::dlib
    )
aq_install_python_test(tests/face_recognition.py)
