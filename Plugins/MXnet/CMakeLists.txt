project(MXnet)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(mxnet QUIET)

if(mxnet_FOUND)
  find_package(OpenBLAS QUIET)
  if(OpenBLAS_FOUND AND OpenBLAS_INCLUDE_DIR)
    include_directories(${OpenBLAS_INCLUDE_DIR})

    if(CUDNN_INCLUDE_DIR)
      LINK_DIRECTORIES(${LINK_DIRS})
      include_directories(${mxnet_INCLUDE_DIRS} 
		${CUDNN_INCLUDE_DIR}
      )
      file(GLOB_RECURSE src "src/*.cpp")
      file(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
      add_library(MXnet SHARED ${src} ${hdr})
      RCC_LINK_LIB(MXnet 
        mxnet
	    ${OpenBLAS_LIB}
        ${CUDNN_LIBRARY}
        Aquila
        MetaObject
      )
      INCLUDE(../PluginTemplate.cmake)
    else(CUDNN_INCLUDE_DIR)
      message(STATUS "-- CUDNN NotFound")
    endif(CUDNN_INCLUDE_DIR)
  else(OpenBLAS_FOUND AND OpenBLAS_INCLUDE_DIR)
    message(STATUS "-- OpenBLAS NotFound")
  endif(OpenBLAS_FOUND AND OpenBLAS_INCLUDE_DIR)
else(mxnet_FOUND)
    message(STATUS "-- MXNet NotFound")
endif(mxnet_FOUND)
