set(DARKNET_ROOT "" CACHE PATH "Path to root install of darknet")

if(WIN32)
rcc_find_library(DARKNET_LIBRARY yolo_cpp_dll.lib
	HINTS ${DARKNET_ROOT}/lib
)

rcc_find_path(DARKNET_INCLUDE yolo_v2_class.hpp HINTS ${DARKNET_ROOT}/include)

else(WIN32)

endif(WIN32)

if(DARKNET_LIBRARY AND DARKNET_INCLUDE)
	find_package(OpenCV QUIET COMPONENTS cudawarping cudaimgproc)
	file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp")
	INCLUDE_DIRECTORIES(
		${DARKNET_INCLUDE}
		${OpenCV_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
		${CUDA_INCLUDE_DIRS}
		${CMAKE_CURRENT_SOURCE_DIR}
		${MetaObject_INCLUDE_DIRS}
	)
	cuda_add_library(darknet SHARED ${src})
	target_link_libraries(darknet 
		aquila_core 
		aquila_types 
		aquila_metatypes 
		metaobject_metaparams
		metaobject_params
		${DARKNET_LIBRARY}
		Core
	)
	aquila_declare_plugin(darknet)
endif()