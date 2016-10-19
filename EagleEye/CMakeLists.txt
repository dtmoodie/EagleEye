
project(EagleEye)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
find_package(Qt5 COMPONENTS Gui Widgets Core OpenGL PrintSupport)
IF(Qt5_FOUND)
    find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d)

    
    find_package(CUDA REQUIRED)

    set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono log log_setup)

    find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

    ADD_DEFINITIONS("-DQt5_FOUND")
    ADD_DEFINITIONS("-DPARAMETERS_USE_UI")
    INCLUDE_DIRECTORIES(
        ${OpenCV_INCLUDE_DIRS}
        ${Parameters_INCLUDE}
        ${Boost_INCLUDE_DIRS}
        "EagleEye/include" 
        ${Qt5Gui_INCLUDE_DIRS}
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5PrintSupport_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS} 
        ${Qt5OpenGL_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${EagleLib_INCLUDE_DIRECTORIES}
    )

    LINK_DIRECTORIES(${LINK_DIRS})
    ADD_DEFINITIONS(-DPARAMETERS_USE_UI)
    file(GLOB sources "src/*.cpp") 
    file(GLOB headers "include/*.h" "include/*.hpp")
    file(GLOB forms "*UIs/*.ui")

    QT5_WRAP_UI(UIS ${forms})
    QT5_WRAP_CPP(MOC ${headers})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS} /FS")

        SET(executable_name "")
        if(WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
            set(executable_name "EagleEye")
        else(WIN32)
            set(executable_name "EagleEye")
        endif(WIN32)

        ADD_EXECUTABLE(${executable_name}
                        ${sources}
                        ${MOC}
                        ${headers_moc}
                        ${headers}
                        ${UIS}
        )
    IF(WIN32)
        FIND_PACKAGE(GstreamerWindows)
        MACRO(APPEND_BIN_DIR ARG1 ARG2)
            IF(EXISTS "${${ARG1}}" AND IS_DIRECTORY "${${ARG1}}")
                LIST(APPEND ${ARG2} ${${ARG1}})
            ELSE()
                MESSAGE("${ARG1} : ${${ARG1}} is not a valid directory")
            ENDIF()
        ENDMACRO(APPEND_BIN_DIR)

        SET(PROJECT_BIN_DIRS_DEBUG PARENT_PATH)
        SET(PROJECT_BIN_DIRS_RELEASE PARENT_PATH)
        SET(PROJECT_BIN_DIRS_RELWITHDEBINFO PARENT_PATH)
		if(OpenCV_FOUND)
           SET(OpenCV_BIN_DIR_DEBUG "${OpenCV_DIR}/x64/vc14/bin" CACHE PATH "" FORCE)
           SET(OpenCV_BIN_DIR_RELEASE "${OpenCV_DIR}/x64/vc14/bin" CACHE PATH "" FORCE)
		endif()
        SET(Qt5_BIN_DIR "${Qt5_DIR}/../../../bin" CACHE PATH "")
        #SET(CAFFE_RELEASE "${Caffe_DIR}/../../bin" CACHE PATH "")
        #SET(CAFFE_DEBUG "${Caffe_DIR}/../../bin" CACHE PATH "")
        SET(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin" CACHE PATH "")
        SET(Boost_BIN_DIR ${Boost_LIBRARY_DIR} CACHE PATH "")
        
        FOREACH(dir ${BIN_DIRS})
            APPEND_BIN_DIR(${dir}_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
            APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
            APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)
        ENDFOREACH(dir ${BIN_DIRS})

        APPEND_BIN_DIR(OpenCV_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
        APPEND_BIN_DIR(OpenCV_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
        APPEND_BIN_DIR(OpenCV_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)

        APPEND_BIN_DIR(Qt5_BIN_DIR PROJECT_BIN_DIRS_DEBUG)
        APPEND_BIN_DIR(Qt5_BIN_DIR PROJECT_BIN_DIRS_RELEASE)
        APPEND_BIN_DIR(Qt5_BIN_DIR PROJECT_BIN_DIRS_RELWITHDEBINFO)

        APPEND_BIN_DIR(CUDA_BIN_DIR PROJECT_BIN_DIRS_DEBUG)
        APPEND_BIN_DIR(CUDA_BIN_DIR PROJECT_BIN_DIRS_RELEASE)
        APPEND_BIN_DIR(CUDA_BIN_DIR PROJECT_BIN_DIRS_RELWITHDEBINFO)

        IF(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")
            LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR_DEBUG})
        ELSE(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")
            IF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
                LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${BOOST_LIBRARYDIR})
            ELSE(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
                MESSAGE("Could not find boost binary dir in Boost_LIBRARY_DIR_DEBUG or BOOST_LIBRARYDIR")
            ENDIF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
        ENDIF(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")

        IF(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
            LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR_RELEASE})
            LIST(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${Boost_LIBRARY_DIR_RELEASE})
        ELSE(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
            IF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
                LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${BOOST_LIBRARYDIR})
                LIST(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${BOOST_LIBRARYDIR})
            ELSE(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
                IF(Boost_LIBRARY_DIR)
                    LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR})
                    LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR})
                ELSE()
                    MESSAGE("Could not find boost binary dir in Boost_LIBRARY_DIR_DEBUG or BOOST_LIBRARYDIR")
                ENDIF()
            ENDIF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
        ENDIF(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")

        APPEND_BIN_DIR(GSTREAMER_gst_BIN_DIR PROJECT_BIN_DIRS_DEBUG)
        APPEND_BIN_DIR(GSTREAMER_gst_BIN_DIR PROJECT_BIN_DIRS_RELEASE)
        APPEND_BIN_DIR(GSTREAMER_gst_BIN_DIR PROJECT_BIN_DIRS_RELWITHDEBINFO)
        
        set(PROJECT_BIN_DIRS_DEBUG ${PROJECT_BIN_DIRS_DEBUG} CACHE PATH "")
        set(PROJECT_BIN_DIRS_RELEASE ${PROJECT_BIN_DIRS_DEBUG} CACHE PATH "")
        set(PROJECT_BIN_DIRS_RELWITHDEBINFO ${PROJECT_BIN_DIRS_DEBUG} CACHE PATH "")

        CONFIGURE_FILE(EagleEye.vcxproj.user.in ${CMAKE_BINARY_DIR}/EagleEye/EagleEye.vcxproj.user @ONLY)
    ENDIF(WIN32)
    ADD_DEFINITIONS(-DBUILD_DIR="${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        add_dependencies(${executable_name} EagleLib frame_grabbers Core)
    GET_TARGET_PROPERTY(LIB_DIR_RELEASE EagleLib IMPORTED_LINK_DEPENDENT_LIBRARIES)

    LINK_DIRECTORIES(${LINK_DIRS})
    SOURCE_GROUP("MOC"  FILES ${MOC} ${UIS})
    SOURCE_GROUP("UIs" FILES ${forms})

    message(STATUS
    "======== EagleEye Reporting =======
      Qt bin dir: ${Qt5_DIR} 
      Boost bin dir: ${Boost_BIN_DIR}
          Bost libs: ${Boost_LIBRARIES}

      CXX Flags: ${CMAKE_CXX_FLAGS}

      Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}

      Release flags: ${CMAKE_CXX_FLAGS_RELEASE}
  
    ")
 
        TARGET_LINK_LIBRARIES(${executable_name}
            Qt5::Gui
            Qt5::Widgets
            Qt5::Core
            Qt5::OpenGL
            Qt5::PrintSupport
            EagleLib
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS} 
    )
    if(UNIX)
        TARGET_LINK_LIBRARIES(${executable_name} dl)
    endif()

ENDIF(Qt5_FOUND)
