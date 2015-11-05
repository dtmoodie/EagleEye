PROJECT(vtkRendering)
find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  GET_TARGET_PROPERTY(vtk_core vtkCommonCore IMPORTED_IMPLIB_RELEASE)
  #GET_TARGET_PROPERTY(vtk_core vtkCommonCore LOCATION)
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

ADD_DEFINITIONS(${DEFS})
LINK_DIRECTORIES(${LINK_DIRS})
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})

ADD_DEFINITIONS(-DPROJECT_INCLUDS="${VTK_INCLUDE_DIRS}")
ADD_DEFINITIONS(-DPROJECT_LIB_DIRS="${vtk_lib_dir}")

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
  ENDIF(MSVC)
ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)
#ADD_MSVC_PRECOMPILED_HEADER("precomp.hpp" "precomp.cpp" src)
CUDA_ADD_LIBRARY(vtkRendering SHARED ${src} ${hdr} ${knl})
INCLUDE(../PluginTemplate.cmake)
ADD_DEPENDENCIES(vtkRendering EagleLib)
TARGET_LINK_LIBRARIES(vtkRendering
  EagleLib
  ${OpenCV_LIBS}
  ${VTK_LIBRARIES}
)