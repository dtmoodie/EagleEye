PROJECT(VLC)
CMAKE_POLICY(SET CMP0020 NEW)
FIND_PACKAGE(LibVLC QUIET)
IF(LIBVLC_FOUND)
    FIND_PACKAGE(Boost REQUIRED COMPONENTS system thread)
    FIND_PACKAGE(CUDA REQUIRED)
    FIND_PACKAGE(OpenCV REQUIRED core videoio)
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
    FIND_PACKAGE(Qt5Network)
    INCLUDE_DIRECTORIES(
            ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${INCLUDE_DIRS}
            ${LIBVLC_INCLUDE_DIR}
            ${INCLUDE_DIRS}

    )

    LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    LINK_DIRECTORIES(${LINK_DIRS})


    FILE(GLOB_RECURSE src "src/*.cpp")
    FILE(GLOB_RECURSE hdr "include/*.h")

    IF(NOT ${LIBVLC_LIBRARY} MATCHES "NOTFOUND")
        CUDA_ADD_LIBRARY(VLC SHARED ${src} ${hdr})
        ADD_DEPENDENCIES(VLC EagleLib)
        RCC_LINK_LIB(VLC
                opencv_core
                opencv_videoio
                EagleLib


                )
        TARGET_LINK_LIBRARIES(VLC ${LINK_LIBS}
                ${LIBVLC_LIBRARY} ${Boost_LIBRARIES}
                ${LIBVLCCORE_LIBRARY})

        SET(LIB_DIRS "C:/Program Files/VideoLan/vlc/sdk/lib")
        GET_PROPERTY(dirs TARGET VLC PROPERTY INCLUDE_DIRECTORIES)
        FOREACH(dir ${INCLUDE_DIRS})
        LIST(REMOVE_ITEM dirs ${dir})
        ENDFOREACH()
        INCLUDE(../PluginTemplate.cmake)


        LIST(REMOVE_DUPLICATES dirs)
        STRING(REGEX REPLACE ";" "+" RCC_INCLUDE_DEPENDENCIES "${dirs}+" )

        ADD_DEFINITIONS(-DPROJECT_INCLUDES_="${RCC_INCLUDE_DEPENDENCIES}")
        ADD_DEFINITIONS(-DPROJECT_LIB_DIRS_="${Qt5_DIR}/../../+${LIB_DIRS}+")
    ENDIF()
ENDIF(LIBVLC_FOUND)
