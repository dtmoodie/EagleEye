
project(EagleEye)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
find_package(Qt5 COMPONENTS Gui Widgets Core OpenGL PrintSupport)
IF(Qt5_FOUND)
	find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d)
	find_package(CUDA REQUIRED)
#IF(WIN32)
#        set(Boost_USE_STATIC_LIBS        ON)
#        set(Boost_USE_MULTITHREADED      ON)
#        set(Boost_USE_STATIC_RUNTIME     OFF)
#ELSE(WIN32)
#        set(Boost_USE_STATIC_LIBS        ON)
#        set(Boost_USE_MULTITHREADED      ON)
#        set(Boost_USE_STATIC_RUNTIME     OFF)
#ENDIF(WIN32)
#ADD_DEFINITIONS(-DBOOST_ALL_NO_LINK)
#ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)
        set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono log log_setup)

	find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


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
	QT5_WRAP_CPP(MOC ${sources} ${headers})

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

	ADD_EXECUTABLE(EagleEye
			${sources}
			${MOC}
			${headers_moc}
			${headers}
			${UIS}
	)
	IF(WIN32)
		SET(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc12/bin")
		SET(Qt5_BIN_DIR "${Qt5_DIR}/../../../bin")
		SET(CAFFE_RELEASE "${Caffe_DIR}/bin/Release")
		SET(CAFFE_DEBUG "${Caffe_DIR}/bin/Debug")
		SET(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
		SET(Boost_BIN_DIR ${Boost_LIBRARY_DIR})
		CONFIGURE_FILE(EagleEye.vcxproj.user.in ${CMAKE_BINARY_DIR}/EagleEye/EagleEye.vcxproj.user @ONLY)
	ENDIF(WIN32)
	ADD_DEFINITIONS(-DBUILD_DIR="${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
	add_dependencies(EagleEye EagleLib)
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
 
	TARGET_LINK_LIBRARIES(EagleEye
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
		TARGET_LINK_LIBRARIES(EagleEye dl)
	endif()

ENDIF(Qt5_FOUND)
