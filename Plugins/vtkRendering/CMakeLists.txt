PROJECT(vtkRendering)

find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  GET_TARGET_PROPERTY(vtk_core vtkCommonCore IMPORTED_IMPLIB_RELEASE)
  GET_FILENAME_COMPONENT(vtk_lib_dir ${vtk_core} DIRECTORY)
  LINK_DIRECTORIES(${vtk_lib_dir})
  INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS})

  MESSAGE(info 
  "============ vtk kits ================
  ${VTK_KITS}
  ============ vtk libs =================
  ${VTK_LIBRARIES}
  ${vtk_core}
  ${vtk_lib_dir}")

FIND_PACKAGE(CUDA REQUIRED)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})  

FIND_PACKAGE(Qt5 REQUIRED Core Widgets OpenGL)

FIND_PACKAGE(OpenCV REQUIRED core viz)

ADD_DEFINITIONS(${DEFS})
LINK_DIRECTORIES(${LINK_DIRS})
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)
#ADD_DEFINITIONS(-DPROJECT_INCLUDS="${VTK_INCLUDE_DIRS}")
#ADD_DEFINITIONS(-DPROJECT_LIB_DIRS="${vtk_lib_dir}")

SET(VTK_BIN_DIR "" CACHE PATH "")
SET(VTK_BIN_DIR_DBG ${VTK_BIN_DIR} CACHE PATH "" FORCE)
SET(VTK_BIN_DIR_OPT ${VTK_BIN_DIR} CACHE PATH "" FORCE)
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
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")

ADD_LIBRARY(vtkRendering SHARED ${src} ${hdr} ${knl})

ADD_DEPENDENCIES(vtkRendering EagleLib)
RCC_LINK_LIB(vtkRendering
  EagleLib
  ${OpenCV_LIBS}
  ${VTK_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
)
INCLUDE(../PluginTemplate.cmake)
