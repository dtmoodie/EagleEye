project(frame_grabber_openni2)

file(GLOB_RECURSE src "src/*.h" "src/*.cpp")
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OpenNI2 QUIET)
if(OpenNI2_FOUND)
  include_directories(${OPENNI2_INCLUDE_DIR})
  
  add_library(frame_grabber_openni2 SHARED ${src})
  target_link_libraries(frame_grabber_openni2 EagleLib)
  add_dependencies(frame_grabber_openni2 EagleLib)
  RCC_LINK_LIB(frame_grabber_openni2 ${OPENNI2_LIBRARY})
  INCLUDE(../PluginTemplate.cmake)
endif()