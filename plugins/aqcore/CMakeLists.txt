PROJECT(aqcore)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
  CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)
  include_directories(${CUDA_TOOLKIT_INCLUDE})


find_package(OpenCV 3.0 QUIET 
  COMPONENTS 
    core 
    imgproc 
    highgui 
    cudaarithm 
    cudaimgproc 
    cudafilters 
    cudawarping 
    cudafeatures2d 
    cudaoptflow 
    cudacodec 
    cudastereo 
    videoio 
    imgcodecs 
    cudaobjdetect 
    objdetect
  )

set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
    ocv_check_modules(FFMPEG libavcodec libavformat libavutil libswscale)
    ocv_check_modules(FFMPEG_libavresample libavresample)
    if(FFMPEG_libavresample_FOUND)
        add_definitions(-DHAVE_FFMPEG)
    endif()
endif()

ADD_DEFINITIONS(${DEFS})
LINK_DIRECTORIES(${LINK_DIRS})

INCLUDE_DIRECTORIES(
    SYSTEM
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(${MetaObject_INCLUDE_DIRS})


file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp")
if(MO_HAVE_CUDA)
    IF(UNIX)
      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
      set(CUDA_NVCC_FLAGS "-std=c++11;${CUDA_NVCC_FLAGS}")
    ENDIF()
    file(GLOB_RECURSE knl "*.cu")
    cuda_add_library(aqcore SHARED ${src} ${hdr} ${knl})
else(MO_HAVE_CUDA)
    add_library(aqcore SHARED ${src} ${hdr})
endif(MO_HAVE_CUDA)


target_link_libraries(aqcore 
    aquila_core
    aquila_utilities
    aquila_serialization
    aquila_types
    aquila_gui

    ${OpenCV_LIBS}
    ${CUDA_cublas_LIBRARY}
    metaobject_core
    metaobject_object
    metaobject_params
)

aquila_declare_plugin(aqcore)
set_plugin_available(aqcore)

