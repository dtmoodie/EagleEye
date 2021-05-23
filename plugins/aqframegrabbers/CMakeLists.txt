PROJECT(aqframegrabbers)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})
  link_directories(${BOOST_LIBRARYDIR})
  link_directories(${Boost_LIBRARY_DIR_DEBUG})
  link_directories(${Boost_LIBRARY_DIR_RELEASE})
  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(OpenCV QUIET COMPONENTS core imgcodecs cudacodec cudaarithm videoio)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)


ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")
if(MO_HAVE_CUDA)
    file(GLOB_RECURSE knl "src/*.cu")
    cuda_add_library(aqframegrabbers SHARED ${src} ${hdr} ${knl})
    target_link_libraries(aqframegrabbers ${CUDA_cublas_LIBRARY})
else(MO_HAVE_CUDA)
    add_library(aqframegrabbers SHARED ${src} ${hdr})
endif(MO_HAVE_CUDA)

target_link_libraries(aqframegrabbers
    aquila_core
    metaobject_core 
    metaobject_object 
    metaobject_params 
    metaobject_metaparams
    ${OpenCV_LIBS}
    aquila_types
)


# ------------- tests
if(BUILD_TESTS)
    add_subdirectory("tests")
endif()

metaobject_declare_plugin(aqframegrabbers)
set_plugin_available(aqframegrabbers)

set(FRAMEGRABBERS_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)

aq_install_python_test(tests/test_directory_grabber.py)
aq_install_python_test(tests/test_html5_grabber.py)
