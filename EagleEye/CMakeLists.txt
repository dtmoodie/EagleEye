cmake_minimum_required(VERSION 2.8)
project(EagleEye)
find_package(Qt5 COMPONENTS Gui Widgets Core OpenGL PrintSupport)
find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d)
find_package(CUDA REQUIRED)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("EagleEye/include")
INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Core_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5PrintSupport_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5OpenGL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${EagleLib_INCLUDE_DIRECTORIES})

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

add_dependencies(EagleEye EagleLib)
SOURCE_GROUP("MOC"  FILES ${MOC} ${UIS})
SOURCE_GROUP("UIs" FILES ${forms})


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

