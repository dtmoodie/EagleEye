project(point_clouds)
find_package(vtk QUIET)
find_package(PCL QUIET)
  if(PCL_FOUND)
    add_definitions(-DHAVE_PCL)
    INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
  endif()
 
  
set(Plugin_point_clouds_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "")
include_directories("include")

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp" "src/*.hpp" "src/*.h")

add_library(point_clouds SHARED ${src} ${knl} ${hdr})

add_dependencies(point_clouds Aquila)

RCC_LINK_LIB(point_clouds
  Aquila
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

INCLUDE(../PluginTemplate.cmake)
  
