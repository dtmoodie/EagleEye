cmake_minimum_required(VERSION 2.8)
find_package(Caffe)

include_directories(
 ${Caffe_INCLUDE_DIRS}
 ${EagleLib_INCLUDE_DIRECTORIES}
)

message(STATUS
" ===================== Caffe Plugin ========================
  Caffe found: ${Caffe_FOUND}
  Caffe Include: ${Caffe_INCLUDE_DIRS}

  Caffe libs: ${Caffe_LIBRARIES}

  EagleLib Includes: ${EagleLib_INCLUDE_DIRECTORIES}
  ===========================================================")

if(${Caffe_FOUND})

add_library(Caffe SHARED Caffe.h Caffe.cpp)
add_dependencies(Caffe EagleLib)
target_link_libraries(Caffe EagleLib ${Caffe_LIBRARIES})

set_target_properties(Caffe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}/Plugins
)
endif()
