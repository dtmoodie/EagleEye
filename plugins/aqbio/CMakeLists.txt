PROJECT(aqbio)

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    set_plugin_unavailable(aqbio)
    set_plugin_status(aqbio "Unsupported gcc version ${CMAKE_CXX_COMPILER_VERSION}")
    return()
endif()

find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc)

file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(aqbio SHARED ${src} ${hdr})

target_link_libraries(aqbio 
    aquila_core
    aquila_utilities
    aquila_serialization
    aquila_types
    ${OpenCV_LIBS}
    aqcore
)

aquila_declare_plugin(aqbio)
set_plugin_available(aqbio)
aq_install_python_test(scripts/activated_bovine.py)
set_plugin_status(aqbio "")
