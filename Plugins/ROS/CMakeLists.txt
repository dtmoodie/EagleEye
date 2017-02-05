project(ROS)
find_package(catkin COMPONENTS
  std_msgs
  roscpp
  cv_bridge
)

if(catkin_FOUND)
  file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.cu")
  include_directories(${catkin_INCLUDE_DIRS})
  cuda_add_library(ROS ${src})
  RCC_LINK_LIB(ROS ${catkin_LIBRARIES} MetaObject EagleLib)
  INCLUDE(../PluginTemplate.cmake)
endif()
