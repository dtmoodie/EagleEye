project(ros)

find_package(catkin QUIET
    COMPONENTS
        std_msgs
        roscpp
)

if(catkin_FOUND)
    file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.cu")
    include_directories(
        ${catkin_INCLUDE_DIRS}
        "/opt/ros/noetic/include"
    )
    find_package(OpenCV REQUIRED COMPONENTS cudaimgproc)
    cuda_add_library(aqros ${src})
    target_link_libraries(aqros
        ${catkin_LIBRARIES}
        aquila_core
        aquila_utilities
        metaobject_serialization
        aquila_types
        ${OpenCV_LIBS}
    )
    metaobject_declare_plugin(aqros)
    set_plugin_available(aqros)
    set_plugin_status(aqros "ROS found: catkin include dir:\"${catkin_INCLUDE_DIRS}\" catkin libraries: \"${catkin_LIBRARIES}\"")
else()
    set_plugin_unavailable(aqros)
    set_plugin_status(aqros "ros not found")
endif()
