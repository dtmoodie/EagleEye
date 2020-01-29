project(ros)

find_package(catkin QUIET COMPONENTS
  std_msgs
  roscpp
)

if(catkin_FOUND)
  file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.cu")
  include_directories(
      ${catkin_INCLUDE_DIRS}
      "/opt/ros/kinetic/include"
    )
  find_package(OpenCV REQUIRED COMPONENTS cudaimgproc)
  cuda_add_library(aqros ${src})
  RCC_LINK_LIB(aqros ${catkin_LIBRARIES} aquila_core aquila_utilities metaobject_serialization aquila_types ${OpenCV_LIBS})
  aquila_declare_plugin(aqros)
  set_plugin_available(aqros)
  set_plugin_status(aqros "ROS found: ${catkin_INCLUDE_DIRS}")
else()
    set_plugin_unavailable(aqros)
    set_plugin_status(aqros "ros not found")
endif()
