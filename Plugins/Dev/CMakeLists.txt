PROJECT(Dev)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})
find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo superres)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
ADD_DEFINITIONS(${DEFS})

INCLUDE_DIRECTORIES(
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}

)

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp" "src/*.cuh")

IF(WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
ENDIF()

cuda_add_library(Dev SHARED ${src} ${hdr} ${knl})

add_dependencies(Dev Aquila)
RCC_LINK_LIB(Dev
    ${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
    Aquila
    MetaParameters
)

INCLUDE(../PluginTemplate.cmake)

