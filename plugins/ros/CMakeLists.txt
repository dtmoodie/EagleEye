project(ros)

find_package(catkin QUIET COMPONENTS
  std_msgs
  roscpp
  cv_bridge
)

if(catkin_FOUND)
  file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.cu")
  include_directories(${catkin_INCLUDE_DIRS})
  find_package(OpenCV REQUIRED COMPONENTS cudaimgproc)
  cuda_add_library(aqros ${src})
  RCC_LINK_LIB(aqros ${catkin_LIBRARIES} aquila_core aquila_utilities metaobject_serialization aquila_types ${OpenCV_LIBS} aquila_metatypes)
  aquila_declare_plugin(aqros)
endif()
