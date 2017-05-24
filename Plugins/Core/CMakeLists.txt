PROJECT(Core)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)
  include_directories(${CUDA_TOOLKIT_INCLUDE})


find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo videoio)
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
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MetaObject_INCLUDE_DIRS}
)

file(GLOB_RECURSE knl "*.cu")
file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-std=c++11;--expt-relaxed-constexpr;${CUDA_NVCC_FLAGS}")
cuda_add_library(Core SHARED ${src} ${hdr} ${knl})
ocv_add_precompiled_header_to_target(Core nodes/precompiled.hpp)

RCC_LINK_LIB(Core aquila_core
    aquila_utilities
    aquila_serialization
    aquila_types
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
    metaobject_core
    metaobject_object
    metaobject_params
)

INCLUDE(../PluginTemplate.cmake)

