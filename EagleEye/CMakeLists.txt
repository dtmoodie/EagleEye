
project(EagleEye)
if(WITH_QT AND Qt5_FOUND)

    find_package(Qt5 QUIET COMPONENTS Gui Widgets Core OpenGL PrintSupport)
    if(NOT BUILD_DEPENDENCIES)
        find_package(nodeeditor QUIET)
    endif()
    if(NOT nodeeditor_FOUND)
	    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/")
	    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/")
	    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/")

	    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/")
	    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/")
	    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/")

	    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/")
	    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/")
	    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/")
	
        add_subdirectory(dependencies/nodeeditor)
        set(nodeeditor_FOUND TRUE)
    endif()

    IF(Qt5_FOUND AND nodeeditor_FOUND)
        if(NOT TARGET aquila_core)
            find_package(Aquila REQUIRED COMPONENTS core)
        endif()
        find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d)

        find_package(CUDA REQUIRED)

        ADD_DEFINITIONS("-DQt5_FOUND")
        INCLUDE_DIRECTORIES(
            SYSTEM
            ${OpenCV_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
            ${Qt5Gui_INCLUDE_DIRS}
            ${Qt5Core_INCLUDE_DIRS}
            ${Qt5PrintSupport_INCLUDE_DIRS}
            ${Qt5Widgets_INCLUDE_DIRS}
            ${Qt5OpenGL_INCLUDE_DIRS}
        )

        INCLUDE_DIRECTORIES(
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}
            ${Aquila_INCLUDE_DIRECTORIES}
        )

        file(GLOB sources "src/*.cpp")
        file(GLOB headers "include/*.h" "include/*.hpp")
        file(GLOB forms "*UIs/*.ui")

        QT5_WRAP_UI(UIS ${forms})
        QT5_WRAP_CPP(MOC ${headers})
	    link_directories(${Boost_LIBRARY_DIR})
	    link_directories(${Boost_BIN_DIR_DEBUG})
	    link_directories(${Boost_LIBRARY_DIR_RELEASE})
	
        ADD_EXECUTABLE(EagleEye
            ${sources}
            ${MOC}
            ${headers_moc}
            ${headers}
            ${UIS}
        )
        IF(WIN32)
            FIND_PACKAGE(GstreamerWindows)
        ENDIF(WIN32)
	
        LINK_DIRECTORIES(${LINK_DIRS})
        SOURCE_GROUP("MOC"  FILES ${MOC} ${UIS})
        SOURCE_GROUP("UIs" FILES ${forms})

        if(RCC_VERBOSE_CONFIG)
            message(STATUS
                "======== EagleEye Reporting =======\n"
                "  Qt bin dir: ${Qt5_DIR}\n"
                "  Boost bin dir: ${Boost_BIN_DIR}\n"
                "  Boost libs: ${Boost_LIBRARIES}\n"
                "  CXX Flags: ${CMAKE_CXX_FLAGS} \n"
                "  Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}\n"
                "  Release flags: ${CMAKE_CXX_FLAGS_RELEASE}"
            )
        endif()

        RCC_LINK_LIB(EagleEye
            Qt5::Gui
            Qt5::Widgets
            Qt5::Core
            Qt5::OpenGL
            Qt5::PrintSupport
            aquila_core
            metaobject_qtgui
            ${Boost_TARGETS}
            ${OpenCV_LIBS}
            nodes
            RuntimeObjectSystem
            RuntimeCompiler
        )
        IF(WIN32)
            target_compile_options(EagleEye PUBLIC "/FS")
		    target_compile_options(EagleEye PUBLIC "/bigobj")
            configure_bin_dirs(EagleEye)
            get_target_property(nodes_dll nodes IMPORTED_LOCATION_DEBUG)
		    GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_DEBUG)
		    GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_RELEASE)
		    GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_RELWITHDEBINFO)
            if(nodes_dll)
                get_filename_component(nodes_bin_dir ${nodes_dll} DIRECTORY)
                list(APPEND PROJECT_BIN_DIRS_DEBUG ${nodes_bin_dir})
			    list(APPEND PROJECT_BIN_DIRS_RELEASE ${nodes_bin_dir})
			    list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${nodes_bin_dir})
            endif()
            CONFIGURE_FILE("../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/EagleEye/EagleEye.vcxproj.user @ONLY)
        else()
            TARGET_LINK_LIBRARIES(EagleEye dl)
        endif(WIN32)
    ENDIF(Qt5_FOUND AND nodeeditor_FOUND)
endif(WITH_QT AND Qt5_FOUND)