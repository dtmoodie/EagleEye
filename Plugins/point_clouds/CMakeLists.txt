project(point_clouds)
find_package(vtk QUIET)
find_package(PCL QUIET)

set(Plugin_point_clouds_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "")
include_directories("include;${PCL_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${Aquila_INCLUDE_DIRS})

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "src/*.hpp" "src/*.h")

add_library(point_clouds SHARED ${src} ${knl} ${hdr})

RCC_LINK_LIB(point_clouds
      metaobject_core metaobject_object metaobject_params metaobject_metaparams
      ${OpenCV_LIBS}
      ${PCL_LIBRARIES}
      #${VTK_LIBRARIES}
      #${VTK_LIBS}
      aquila_core
      aquila_types
)

if(PCL_FOUND)
    target_compile_definitions(point_clouds PRIVATE HAVE_PCL)
endif()

if(vtk_FOUND)
    target_compile_definitions(point_clouds PRIVATE HAVE_VTK)
endif()

INCLUDE(../PluginTemplate.cmake)

