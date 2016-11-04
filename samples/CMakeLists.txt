

project(samples)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
    CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono program_options log log_setup)
find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})


find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
  set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})

LINK_DIRECTORIES(${LINK_DIRS})

find_package(CUDA REQUIRED)
  
INCLUDE_DIRECTORIES(
    ${EagleLib_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(SimpleConsole SimpleConsole.cpp)
add_dependencies(SimpleConsole EagleLib)
target_link_libraries(SimpleConsole
    MetaObject
    EagleLib
    ${Boost_LIBRARIES}
    RuntimeObjectSystem
    RuntimeCompiler
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cupti_LIBRARY}
    ${CUDA_nppi_LIBRARY}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_npps_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${PYTHON_LIBRARY}
    instantiations
)
add_subdirectory(PythonInteractive)
IF(WIN32)
  set(PROJECT_BIN_DIRS_DEBUG "")
  set(PROJECT_BIN_DIRS_RELEASE "")
  set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
  MACRO(APPEND_BIN_DIR ARG1 ARG2)
    IF(EXISTS "${${ARG1}}" AND IS_DIRECTORY "${${ARG1}}")
        LIST(APPEND ${ARG2} ${${ARG1}})
    ELSE()
        MESSAGE("${ARG1} : ${${ARG1}} is not a valid directory")
    ENDIF()
  ENDMACRO(APPEND_BIN_DIR)

  FOREACH(dir ${BIN_DIRS})
    APPEND_BIN_DIR(${dir}_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
    APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
    APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)
  ENDFOREACH(dir ${BIN_DIRS})
  CONFIGURE_FILE("../EagleLib/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/samples/SimpleConsole.vcxproj.user @ONLY)
endif(WIN32)
    
    
