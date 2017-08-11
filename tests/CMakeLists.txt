cmake_minimum_required(VERSION 2.8)

project(Tests)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
    CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

cmake_policy(SET CMP0020 NEW)

set(BOOST_REQUIRED_MODULES program_options thread)
find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(
    ${Aquila_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${PPLX_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${LINK_DIRS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

FILE(GLOB CPP_TESTS "*.cpp")
GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_DEBUG)
GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_RELEASE)
GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_RELWITHDEBINFO)
FOREACH( test ${CPP_TESTS} )
  get_filename_component( name ${test} NAME_WE )
  add_executable(${name} ${test})
  target_link_libraries(${name} ${LINK_LIBS} ${Boost_LIBRARIES} ${Aquila_LIBRARIES})
  if(WITH_PYTHON AND HAVE_PYTHON)
    target_link_libraries(${name} ${PYTHON_LIBRARY})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER Tests/EagleEye)
  IF(WIN32)
    CONFIGURE_FILE("../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/tests/${name}.vcxproj.user @ONLY)
  ENDIF(WIN32)
ENDFOREACH( test CPP_TESTS )

