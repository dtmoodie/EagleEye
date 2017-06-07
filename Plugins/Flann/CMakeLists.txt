PROJECT(Flann)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
  CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})
find_package(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON)

find_package(OpenCV 3.0 QUIET COMPONENTS core)
#set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

FIND_PACKAGE(Flann QUIET)

IF(Aquila_FLANN_LIBRARY_DEBUG AND Aquila_FLANN_CUDA_LIBRARY_DEBUG)
    FILE(GLOB_RECURSE src "src/*.cpp")
    FILE(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
    FILE(GLOB_RECURSE knl "src/*.cu")

    LINK_DIRECTORIES(${LINK_DIRS_DEBUG})
    LINK_DIRECTORIES(${LINK_DIRS_RELEASE})
    INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Aquila_FLANN_INCLUDE_DIR}
        )
    GET_FILENAME_COMPONENT(FLANN_LIB_DIR "${Aquila_FLANN_LIBRARY_DEBUG}" PATH)

    ADD_LIBRARY(Flann SHARED ${src} ${hdr})


    RCC_LINK_LIB(Flann aquila_core
        metaobject_core metaobject_object metaobject_params metaobject_metaparams
        ${OpenCV_LIBS}
        ${CUDA_CUBLAS_LIBRARIES}
		aquila_types
		aquila_metatypes
    )

    target_link_libraries(Flann debug;${Aquila_FLANN_CUDA_LIBRARY_DEBUG};
                                debug;${Aquila_FLANN_LIBRARY_DEBUG};
                                optimized;${Aquila_FLANN_CUDA_LIBRARY_RELEASE};
                                optimized;${Aquila_FLANN_LIBRARY_RELEASE}
    )

    #INCLUDE(../PluginTemplate.cmake)
    aquila_declare_plugin(Flann)
ELSE()
MESSAGE("-- Flann:\t\tNotFound")
ENDIF()

