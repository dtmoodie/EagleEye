PROJECT(Segmentation)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
    CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
CMAKE_POLICY(SET CMP0020 NEW)

option(BUILD_FASTMS FALSE)

IF(NOT WIN32)
    set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
ENDIF(NOT WIN32)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo cudabgsegm)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
if(BUILD_FASTMS)
    ADD_SUBDIRECTORY(fastms)
    set_target_properties(fastms PROPERTIES FOLDER Dependencies)
    set_target_properties(fastms
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

INCLUDE_DIRECTORIES(
    SYSTEM
    ${INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${LINK_DIRS})
file(GLOB src "*.cpp")
file(GLOB hdr "*.h" "*.hpp" "*.cuh")
file(GLOB knl "*.cu")

cuda_add_library(Segmentation SHARED ${src} ${hdr})

RCC_LINK_LIB(Segmentation
    metaobject_core
    metaobject_object
    metaobject_params
    metaobject_metaparams
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    aquila_types
    aquila_metatypes
    aquila_utilities
)


IF(TARGET fastms)
    add_dependencies(Segmentation fastms)
    RCC_LINK_LIB(Segmentation fastms ${fastms_LIBS})
    TARGET_LINK_LIBRARIES(Segmentation fastms)
    add_definitions(-DFASTMS_FOUND)
    add_definitions(-DFASTMS_INCLUDE=${fastms_INCLUDE_DIRECTORIES})
    INCLUDE_DIRECTORIES(${fastms_INCLUDE_DIRECTORIES})

ENDIF()
#INCLUDE(../PluginTemplate.cmake)
aquila_declare_plugin(Segmentation)
