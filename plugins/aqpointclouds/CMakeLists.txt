project(aqpointclouds)
#find_package(vtk QUIET)
#find_package(PCL QUIET)

# https://github.com/PointCloudLibrary/pcl/blob/master/cmake/pcl_utils.cmake
macro(REMOVE_VTK_DEFINITIONS)
    get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(_vtk_definitions)
    foreach(_item ${_dir_defs})
        if(_item MATCHES "vtk*")
            list(APPEND _vtk_definitions -D${_item})
        endif()
    endforeach()
    remove_definitions(${_vtk_definitions})
endmacro(REMOVE_VTK_DEFINITIONS)

include_directories("include;${PCL_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${Aquila_INCLUDE_DIRS})

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "src/*.hpp" "src/*.h")

REMOVE_VTK_DEFINITIONS()
cuda_add_library(aqpointclouds SHARED ${src} ${knl} ${hdr})

target_link_libraries(aqpointclouds
      metaobject_core metaobject_object metaobject_params metaobject_metaparams
      ${OpenCV_LIBS}
      ${PCL_LIBRARIES}
      aquila_core
      aquila_types
      aqcore
)

if(PCL_FOUND)
    target_compile_definitions(aqpointclouds PRIVATE HAVE_PCL)
endif()

if(vtk_FOUND)
    target_compile_definitions(aqpointclouds PRIVATE HAVE_VTK)
endif()

aquila_declare_plugin(aqpointclouds)
set_plugin_available(aqpointclouds)
