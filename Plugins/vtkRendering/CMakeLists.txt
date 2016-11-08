PROJECT(vtkRendering)
CMAKE_POLICY(SET CMP0020 NEW)
find_package(VTK QUIET
                 COMPONENTS 
                 vtkInteractionStyle 
				 vtkInteractionWidgets
                 vtkRenderingLOD 
                 vtkIOPLY 
                 vtkFiltersTexture 
                 vtkRenderingFreeType 
                 vtkIOExport 
                 vtkRenderingOpenGL2 
                 vtkRenderingQt
                 vtkGUISupportQt
                 vtkGUISupportQtOpenGL
                 vtkRenderingAnnotation
				 vtkRenderingVolume
				 vtkRenderingVolumeOpenGL2
                 )
if(VTK_FOUND)

  include(${VTK_USE_FILE})
  GET_TARGET_PROPERTY(vtk_core vtkCommonCore IMPORTED_IMPLIB_RELEASE)
  GET_FILENAME_COMPONENT(vtk_lib_dir ${vtk_core} DIRECTORY)
  LINK_DIRECTORIES(${vtk_lib_dir})
  INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS})

  MESSAGE(STATUS  
  "============ vtk libs =================
  ${VTK_LIBRARIES}
  ${vtk_core}
  ${vtk_lib_dir}")

FIND_PACKAGE(CUDA REQUIRED)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})  

FIND_PACKAGE(Qt5 REQUIRED Core Widgets OpenGL)

FIND_PACKAGE(OpenCV QUIET COMPONENTS core viz)

find_package(PCL QUIET COMPONENTS visualization)
  if(PCL_FOUND)
    message("Building with PCL visualization support")
    add_definitions(-DHAVE_PCL)
    include_directories(${PCL_INCLUDE_DIRS})
  endif()
set(link_libs_ 
  EagleLib
  ${OpenCV_LIBS}
  ${VTK_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL)

ADD_DEFINITIONS(${DEFS})
LINK_DIRECTORIES(${LINK_DIRS})
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)
if(Plugin_point_clouds_INCLUDES)
    include_directories(${Plugin_point_clouds_INCLUDES})
    add_definitions(-DHAVE_PLUGIN_POINT_CLOUDS)
    list(APPEND link_libs_ point_clouds)
endif()

get_target_property(VTK_BIN vtkInteractionStyle IMPORTED_LOCATION_DEBUG)
get_filename_component(VTK_BIN_DIR_ "${VTK_BIN}" DIRECTORY)
set(VTK_BIN_DIR ${VTK_BIN_DIR_} CACHE PATH "" FORCE)
set(VTK_BIN_DIR_DBG ${VTK_BIN_DIR_} CACHE PATH "" FORCE)
set(VTK_BIN_DIR_OPT ${VTK_BIN_DIR_} CACHE PATH "" FORCE)
SET(BIN_DIRS "${BIN_DIRS};VTK" CACHE STRING "" FORCE)

FILE(GLOB_RECURSE src "src/*.cpp")
FILE(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE knl "src/*.cu")

MACRO(ADD_MSVC_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
  IF(MSVC)
    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
    SET(PrecompiledBinary "${CMAKE_CURRENT_BINARY_DIR}/${PrecompiledBasename}.pch")
    SET(Sources ${${SourcesVar}})

    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
    SET_SOURCE_FILES_PROPERTIES(${Sources}
                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledHeader}\" /FI\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_DEPENDS "${PrecompiledBinary}")  
    # Add precompiled header to SourcesVar
    LIST(APPEND ${SourcesVar} ${PrecompiledSource})
  ELSE(MSVC)
  ENDIF(MSVC)
ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)

ADD_LIBRARY(vtkRendering SHARED ${src} ${hdr} ${knl})

ADD_DEPENDENCIES(vtkRendering EagleLib)
RCC_LINK_LIB(vtkRendering
  ${link_libs_}
)
if(Plugin_point_clouds_INCLUDES)
  add_dependencies(vtkRendering point_clouds)
endif()
IF(WIN32)
  RCC_LINK_LIB(vtkRendering opengl32)
ELSE(WIN32)

ENDIF(WIN32)

INCLUDE(../PluginTemplate.cmake)
endif(VTK_FOUND)