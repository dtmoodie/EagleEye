cmake_minimum_required(VERSION 2.8)

PROJECT(PCL_bridge)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
	CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

FIND_PACKAGE(Qt5 REQUIRED Core Widgets)

FIND_PACKAGE(CUDA REQUIRED)
	SET(CUDA_VERBOSE_BUILD ON)
	SET(CUDA_SEPARABLE_COMPILATION ON)
	INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
	set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

FIND_PACKAGE(PCL COMPONENTS common visualization QUIET)
macro(REMOVE_VTK_DEFINITIONS)
    get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(_vtk_definitions)
    foreach(_item ${_dir_defs})
        if(_item MATCHES "vtk*")
            list(APPEND _vtk_definitions -D${_item})
        endif()
    endforeach()
    remove_definitions(${_vtk_definitions})
endmacro(REMOVE_VTK_DEFINITIONS)
IF(PCL_FOUND)
	FIND_PACKAGE(VTK QUIET)
	if(VTK_FOUND)
		LINK_DIRECTORIES(${LINK_DIRS})

		include_directories(${PCL_INCLUDE_DIRS})
		link_directories(${PCL_LIBRARY_DIRS})
		
		INCLUDE_DIRECTORIES(
				${EagleLib_INCLUDE_DIRECTORIES}
				${OpenCV_INCLUDE_DIRS}
				${Boost_INCLUDE_DIRS}
			)

		SET(PCL_BIN_DIR_DEBUG "${PCL_DIR}/../bin" CACHE PATH "Path to PCL debug binaries")
		SET(PCL_BIN_DIR_RELEASE "${PCL_DIR}/../bin" CACHE PATH "Path to PCL release binaries")

		
		SET(VTK_BIN_DIR "${VTK_LIBRARY_DIRS}/../bin" CACHE PATH "Path to VTK binaries")
		
		FILE(GLOB_RECURSE SRC "*.cpp")
		FILE(GLOB_RECURSE HDR "*.h")
		#FILE(GLOB_RECURSE knl "*.cu")
        REMOVE_VTK_DEFINITIONS()
		cuda_add_library(PCL_bridge SHARED ${SRC} ${HDR} ${knl})
		add_dependencies(PCL_bridge EagleLib)

		RCC_LINK_LIB(PCL_bridge
			EagleLib
			${CUDA_LIBRARIES}
		)
		
		TARGET_LINK_LIBRARIES(PCL_bridge ${EagleLib_LIBRARIES} ${PCL_COMMON_LIBRARIES}
			${PCL_VISUALIZATION_LIBRARIES})

		SET(pcl_bridge_file "")
		IF(WIN32)
			#SET(pcl_bridge_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PCL_bridge.dir/PCL_bridge_generated_pcl_bridge.cu.obj.cmake")
		ELSE(WIN32)
			#SET(pcl_bridge_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PCL_bridge.dir/PCL_bridge_generated_pcl_bridge.cu.o.cmake")
		ENDIF(WIN32)
		MESSAGE("Correcting file ${pcl_bridge_file}")
		
		#FILE(READ ${pcl_bridge_file} config_file)
		#STRING(REPLACE "(vtkRenderingContextOpenGL)" "" config_file "${config_file}")
		#STRING(REPLACE "(vtkRenderingVolumeOpenGL)" "" config_file "${config_file}")
		#STRING(REPLACE "(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL)" "" config_file "${config_file}")
		#FILE(WRITE ${pcl_bridge_file} "${config_file}")

		INCLUDE(../PluginTemplate.cmake)
	ENDIF(VTK_FOUND)
ENDIF(PCL_FOUND)

