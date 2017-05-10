

project(samples)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
    CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")

set(link_libs "")

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono program_options log log_setup signals)
find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})
  list(APPEND link_libs ${Boost_LIBRARIES})

find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
  set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)


find_package(Aquila REQUIRED)
find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})

LINK_DIRECTORIES(${LINK_DIRS})

find_package(CUDA REQUIRED)
if(WITH_WT)
find_package(Wt QUIET)
  if(Wt_FOUND)
    include_directories(${Wt_INCLUDE_DIR})
    add_definitions(-DHAVE_WT)
    if(WIN32)
      SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
      find_library(Wt_bin NAMES wt PATHS ${Wt_DIR} PATH PATH_SUFFIXES bin)
      get_filename_component(Wt_bin_dir "${Wt_bin}" DIRECTORY)
      set(Wt_BIN_DIR_ ${Wt_bin_dir})
    endif(WIN32)
    set(Wt_BIN_DIR ${Wt_BIN_DIR_} CACHE PATH "" FORCE)
    set(Wt_BIN_DIR_DBG ${Wt_BIN_DIR_} CACHE PATH "" FORCE)
    set(Wt_BIN_DIR_OPT ${Wt_BIN_DIR_} CACHE PATH "" FORCE)
    set(BIN_DIRS "${BIN_DIRS};Wt" CACHE STRING "" FORCE)
  endif()
endif(WITH_WT)

INCLUDE_DIRECTORIES(
    ${Aquila_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
file(GLOB src "*.cpp")
file(GLOB hdr "*.h" "*.hpp")

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(SimpleConsole ${src} ${hdr})
add_dependencies(SimpleConsole Aquila)
target_link_libraries(SimpleConsole
    MetaObject
    Aquila
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
    ${link_libs}
    MetaParameters
    aquila_types
    aquila_metatypes
)
if(MetaObject_WITH_WT AND Wt_FOUND)
    target_compile_definitions(SimpleConsole
        PRIVATE "-DHAVE_WT"
    )
    target_include_directories(SimpleConsole
        PRIVATE ${Wt_INCLUDE_DIR}
    )
endif()

if(WITH_PYTHON AND HAVE_PYTHON)
  target_link_libraries(SimpleConsole ${PYTHON_LIBRARY})
endif(WITH_PYTHON AND HAVE_PYTHON)

if(Wt_FOUND AND WITH_WT)
    target_link_libraries(SimpleConsole
        debug;${Wt_HTTP_DEBUG_LIBRARY}
        debug;${Wt_DEBUG_LIBRARY}
        optimized;${Wt_LIBRARY}
        optimized;${Wt_HTTP_LIBRARY}
    )
endif(Wt_FOUND AND WITH_WT)

INSTALL(TARGETS SimpleConsole
        RUNTIME DESTINATION bin
)

set_property(TARGET SimpleConsole PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

IF(WIN32)
  set(PROJECT_BIN_DIRS_DEBUG "")
  set(PROJECT_BIN_DIRS_RELEASE "")
  set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
  MACRO(APPEND_BIN_DIR ARG1 ARG2)
    IF(EXISTS "${${ARG1}}" AND IS_DIRECTORY "${${ARG1}}")
        LIST(APPEND ${ARG2} ${${ARG1}})
    ELSE()
        if(RCC_VERBOSE_CONFIG)
          MESSAGE("${ARG1} : ${${ARG1}} is not a valid directory")
        endif()
    ENDIF()
  ENDMACRO(APPEND_BIN_DIR)

  FOREACH(dir ${BIN_DIRS})
    APPEND_BIN_DIR(${dir}_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
    APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
    APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)
    APPEND_BIN_DIR(${dir}_BIN_DIR PROJECT_BIN_DIRS_DEBUG)
    APPEND_BIN_DIR(${dir}_BIN_DIR PROJECT_BIN_DIRS_RELEASE)
    APPEND_BIN_DIR(${dir}_BIN_DIR PROJECT_BIN_DIRS_RELWITHDEBINFO)
  ENDFOREACH(dir ${BIN_DIRS})
  list(APPEND PROJECT_BIN_DIRS_DEBUG ${Aquila_BIN_DIRS})
  if(RCC_VERBOSE_CONFIG)
      message(STATUS "SimpleConsole binary runtime directories:\n ${PROJECT_BIN_DIRS_DEBUG}")
  endif()
  CONFIGURE_FILE("../../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/samples/SimpleConsole/SimpleConsole.vcxproj.user @ONLY)
endif(WIN32)


