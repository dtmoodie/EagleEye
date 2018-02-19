project(aqpointclouds)
find_package(vtk QUIET)
find_package(PCL QUIET)

include_directories("include;${PCL_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${Aquila_INCLUDE_DIRS})

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "src/*.hpp" "src/*.h")

cuda_add_library(aqpointclouds SHARED ${src} ${knl} ${hdr})

RCC_LINK_LIB(aqpointclouds
      metaobject_core metaobject_object metaobject_params metaobject_metaparams
      ${OpenCV_LIBS}
      ${PCL_LIBRARIES}
      aquila_core
      aquila_types
)

if(PCL_FOUND)
    target_compile_definitions(aqpointclouds PRIVATE HAVE_PCL)
endif()

if(vtk_FOUND)
    target_compile_definitions(aqpointclouds PRIVATE HAVE_VTK)
endif()

aquila_declare_plugin(aqpointclouds)
set_plugin_available(aqpointclouds)
