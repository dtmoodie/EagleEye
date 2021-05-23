project(openni2)

file(GLOB_RECURSE src "src/*.h" "src/*.cpp")
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OpenNI2 QUIET)

if(OPENNI2_INCLUDE_DIR)
  find_package(OpenCV QUIET COMPONENTS cudaarithm)
  if(OpenCV_FOUND)
  
      include_directories(${OPENNI2_INCLUDE_DIR})
      add_library(aqopenni2 SHARED ${src})

      target_link_libraries(aqopenni2
          PUBLIC
            aquila_core
            aquila_types
            ${OPENNI2_LIBRARY}
            RuntimeObjectSystem
            RuntimeCompiler
            ${OpenCV_LIBS}
        )
      metaobject_declare_plugin(aqopenni2)
      find_path(OPENNI2_BIN_DIR NAMES OpenNI2.dll PATHS ${OPENNI2_LIB_DIR}/../Redist)


      set_plugin_available(aqopenni2)
      set_plugin_status(aqopenni2 "OpenNI2 found")
  else()
    set_plugin_unavailable(aqopenni2)
    set_plugin_status(aqopenni2 "opencv cudaarithm not found")
  endif()
else()
    set_plugin_unavailable(aqopenni2)
    set_plugin_status(aqopenni2 "OpenNI2 no found")
endif()
