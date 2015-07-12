
project(EagleEye)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
find_package(Qt5 COMPONENTS Gui Widgets Core OpenGL PrintSupport)
find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d)
find_package(CUDA REQUIRED)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

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
	CONFIGURE_FILE(EagleEye.vcxproj.user.in ${CMAKE_BINARY_DIR}/EagleEye/EagleEye.vcxproj.user @ONLY)
ENDIF(WIN32)
ADD_DEFINITIONS(-DBUILD_DIR="${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
add_dependencies(EagleEye EagleLib)
SOURCE_GROUP("MOC"  FILES ${MOC} ${UIS})
SOURCE_GROUP("UIs" FILES ${forms})

message(STATUS
"======== EagleEye Reporting =======
  Qt bin dir: ${Qt5_DIR}

  
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

