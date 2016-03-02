

project(samples)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
	CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono program_options log log_setup)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})


find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
LINK_DIRECTORIES(${LINK_DIRS})

INCLUDE_DIRECTORIES(
	${EagleLib_INCLUDE_DIRECTORIES}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(SimpleConsole SimpleConsole.cpp)
add_dependencies(SimpleConsole EagleLib)
target_link_libraries(SimpleConsole
    EagleLib
    ${Boost_LIBRARIES}
    RuntimeObjectSystem
    RuntimeCompiler)
