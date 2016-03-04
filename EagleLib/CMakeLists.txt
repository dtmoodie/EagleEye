cmake_minimum_required(VERSION 2.8)
project(EagleLib)
CMAKE_POLICY(SET CMP0020 NEW)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_DEBUG_POSTFIX d)
IF(WIN32)
  add_definitions(-DWIN32)
  add_definitions(-D_WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
	
  SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
  SET(CMAKE_CXX_FLAGS "/EHsc /FC /Zo /d2Zi+ /Zi /MP /Zm512" CACHE STRING "" FORCE)
  SET(CMAKE_CXX_FLAGS_DEBUG "/MDd /Od" CACHE STRING "" FORCE)
  SET(CMAKE_CXX_FLAGS_RELEASE "/MD /Zi /O2 /Ob1 /D NDEBUG" CACHE STRING "" FORCE)
  SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2" CACHE STRING "" FORCE)
  ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)
  ADD_DEFINITIONS(-D_WIN32_WINNT=0x0601)
ELSE(WIN32)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-reorder -Wno-sign-compare -Wl,--no-as-needed -O0")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -g")
  SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin/Plugins" CACHE PATH "Library output path" FORCE)
  SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Executable output path" FORCE)
ENDIF(WIN32)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudabgsegm)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(CUDA REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets Network OpenGL)
cmake_policy(SET CMP0012 NEW)

IF(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib)
	SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/)
ELSE(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib64/cudadevrt)
	SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
ENDIF(WIN32)
SET(LINK_LIBS "")
ADD_DEFINITIONS(-DCVAPI_EXPORTS)
ADD_DEFINITIONS(-DOPENCV_FOUND)
ADD_DEFINITIONS(-DEagleLIB_EXPORTS)
ADD_DEFINITIONS(-DRMT_DLL)
ADD_DEFINITIONS(${Parameters_DEFS})
ADD_DEFINITIONS(-DBUILD_DIR="${CMAKE_BINARY_DIR}")
ADD_DEFINITIONS(-DCVAPI_EXPORTS)
ADD_DEFINITIONS(-DOPENCV_FOUND)
ADD_DEFINITIONS(-DEagleLIB_EXPORTS)
IF(${NV_TOOLS_EXT_FOUND})
  ADD_DEFINITIONS(-DUSE_NVTX)
  list(APPEND LINK_LIBS ${NV_TOOLS_EXT_LIB})
  include_directories(${NV_TOOLS_EXT_INCLUDE_DIR})
  SET(NVTX_BIN_DIR_OPT "${NV_TOOLS_EXT_BIN_DIR}" CACHE PATH "")
  SET(NVTX_BIN_DIR_DBG "${NV_TOOLS_EXT_BIN_DIR}" CACHE PATH "")
  SET(BIN_DIRS "${BIN_DIRS};NVTX" CACHE STRING "" FORCE)
endif()
LIST(APPEND defs_ ${Parameters_DEFS})
LIST(APPEND defs_ -DRMT_USE_CUDA)
LIST(APPEND defs_ -DRMT_USE_OPENGL)
IF(${EagleLib_JETSON})
	LIST(APPEND defs_ "-DJETSON")
ENDIF(${EagleLib_JETSON})
IF(NOT WIN32)
        set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono regex log log_setup atomic)
ENDIF(NOT WIN32)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES}) 

FILE(GLOB_RECURSE src "src/*.cpp" "remotery/lib/*.c")
FILE(GLOB_RECURSE hdr "src/*.h" "src/*.hpp" "remotery/lib/*.h")
FILE(GLOB_RECURSE ext_inc "include/EagleLib/rcc/external_includes")
FILE(GLOB_RECURSE knl "src/*.cu")
ADD_SUBDIRECTORY("parameters")
ADD_SUBDIRECTORY("signals")
    include_directories("signals")
ADD_SUBDIRECTORY("pplx")

INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${Parameters_INCLUDE}
    ${Qt5Gui_INCLUDE_DIRS} 
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Network_INCLUDE_DIRS}
    ${PPLX_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/signals
    ${CMAKE_CURRENT_SOURCE_DIR}/remotery/lib
    )



# for RCC to be able to recompile modules
LIST(GET OpenCV_INCLUDE_DIRS 0 OpenCV_INCLUDE_DIR)
LIST(GET OpenCV_LIB_DIR 0 CV_LIB_DIR)
if(${CV_LIB_DIR} MATCHES NOTFOUND)
 message("OpenCV_LIB_DIR not set, attempting to find library dir")
endif()

SET(IObject_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

SET(RCC_INCLUDE_DEPENDENCIES "${Boost_INCLUDE_DIRS};${OpenCV_INCLUDE_DIR};${CUDA_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/src;${Parameters_INCLUDE};${PPLX_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/signals;${CMAKE_CURRENT_SOURCE_DIR}/remotery/lib")

IF(WIN32)
	SET(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIR};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib/x64;${Parameters_LIB_DIRS_RELEASE};${CMAKE_CURRENT_BINARY_DIR}/../Debug")
	SET(RCC_LIBRARY_DIRS_RELEASE "${Boost_LIBRARY_DIR};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib/x64;${Parameters_LIB_DIRS_RELEASE};${CMAKE_CURRENT_BINARY_DIR}/../Release;")
	SET(RCC_LIBRARY_DIRS_RELWITHDEBINFO "${Boost_LIBRARY_DIR};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib/x64;${Parameters_LIB_DIRS_RELEASE};${CMAKE_CURRENT_BINARY_DIR}/../RelWithDebInfo")
ELSE()
    SET(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIR};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64;${LIBRARY_OUTPUT_PATH};${Parameters_LIB_DIRS_DEBUG}")
    SET(RCC_LIBRARY_DIRS_RELEASE "${Boost_LIBRARY_DIR};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64;${LIBRARY_OUTPUT_PATH};${Parameters_LIB_DIRS_RELEASE}")
ENDIF()

ADD_SUBDIRECTORY("RCC")

INCLUDE_DIRECTORIES(${RCC_INCLUDE_DIRS})


set_target_properties(pplx PROPERTIES FOLDER Dependencies)
set_target_properties(signals PROPERTIES FOLDER Dependencies)
set_target_properties(parameters PROPERTIES FOLDER Dependencies)
set_target_properties(RuntimeCompiler PROPERTIES FOLDER Dependencies)
set_target_properties(RuntimeObjectSystem PROPERTIES FOLDER Dependencies)
IF(NOT WIN32)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
ENDIF()


#
#				Export variables for other projects to use
#
SET(LIB_BUILD_DIR_RELEASE "")
SET(LIB_BUILD_DIR_DEBUG "")

IF(WIN32)
    SET(LIB_BUILD_DIR_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo")
    SET(LIB_BUILD_DIR_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug")
ELSE(WIN32)
    SET(LIB_BUILD_DIR_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release/Plugins")
    SET(LIB_BUILD_DIR_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug/Plugins")
ENDIF(WIN32)

set(EagleLib_INCLUDE_DIRECTORIES "${Parameters_INCLUDE};${CMAKE_CURRENT_SOURCE_DIR}/src;${CMAKE_CURRENT_SOURCE_DIR}/remotery/lib;${CMAKE_CURRENT_SOURCE_DIR}/signals;${Boost_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS};${RCC_INCLUDE_DIRS};${Qt5Gui_INCLUDE_DIRS};${Qt5Widgets_INCLUDE_DIRS};${Qt5Network_INCLUDE_DIRS};${PPLX_INCLUDE_DIRS}" PARENT_SCOPE)
set(EagleLib_LIBRARIES "EagleLib;${Boost_LIBRARIES};${Parameters_LIBRARIES};signals;pplx" PARENT_SCOPE)

set(EagleLib_LIBS "EagleLib;${Parameters_LIBRARIES};signals;pplx")
SET(EagleLib_LIB_DIRS_DEBUG "${LIB_BUILD_DIR_DEBUG};${Boost_LIBRARY_DIR};${Parameters_LIB_DIRS_DEBUG};${CUDA_LIB_DIR}" PARENT_SCOPE)
SET(EagleLib_LIB_DIRS_RELEASE "${LIB_BUILD_DIR_RELEASE};${Boost_LIBRARY_DIR};${Parameters_LIB_DIRS_RELEASE};${CUDA_LIB_DIR}" PARENT_SCOPE)
SET(EagleLib_DEFS "${defs_}" PARENT_SCOPE)

CONFIGURE_FILE(EagleLibConfig.cmake.in ${CMAKE_BINARY_DIR}/EagleLibConfig.cmake @ONLY)
#CONFIGURE_FILE(EagleLibConfig.cmake.in ${CMAKE_INSTALL_PREFIX}/EagleLibConfig.cmake @ONLY)



# Link directories
LINK_DIRECTORIES(${Parameters_LIB_DIRS_DEBUG})
LINK_DIRECTORIES(${Parameters_LIB_DIRS_RELEASE})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
# Reporting section
IF(Parameters_UI_qt)
	message(STATUS "Parameters built with QT support, so linking into EagleLib")
ENDIF()
message(STATUS
"================ EagleLib ===============
    Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}
    OpenCV found at: ${OpenCV_CONFIG_PATH}
    NVCC Flags: ${CUDA_NVCC_FLAGS}
    Jetson: ${EAGLE_LIB_JETSON}
    Bulid Type: ${CMAKE_BUILD_TYPE}

    Includes: ${EagleLib_INCLUDE_DIRECTORIES}

    Libs: ${EagleLib_LIBRARIES}

    Lib dirs:
	${EagleLib_LIB_DIRS_DEBUG}

    ${EagleLib_LIB_DIRS_RELEASE}

    Definitions: ${defs_}


++++++++++ RCC++ Defines ++++++++++++++++++
    OpenCV include dir: ${OpenCV_INCLUDE_DIR}
    Boost library dir: ${Boost_LIBRARY_DIRS}
    OpenCV library dir: ${CV_LIB_DIR}
    Cuda include dir: ${CUDA_INCLUDE_DIRS}
    NVCC: ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc
    Cuda RT libs: ${CUDA_LIBRARIES}
    Cuda Dev RT libs: ${CUDA_DEVRT}
    Cuda root dir: ${CUDA_TOOLKIT_ROOT_DIR}
    Binary Dir: ${CMAKE_BINARY_DIR}

    Boost LIBS: ${Boost_LIBRARIES}

    PCL_found: ${PCL_FOUND}
    PCL_LIB_DIR: ${PCL_LIBRARY_DIRS}
    Eigen inc: ${EIGEN3_INCLUDE_DIR}
++++++++++++ Parameters ++++++++++++++++++++
	Includes: ${Parameters_INCLUDE}
	UI: ${Parameters_UI_qt}
")

ADD_DEFINITIONS(${defs_})


macro(GroupSources curdir)
file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)
    foreach(child ${children})
		if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
            GroupSources(${curdir}/${child})
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
        endif()
   endforeach()
endmacro()
 
GroupSources(include)
GroupSources(src)
cuda_add_library(EagleLib SHARED
	${src}  
	${hdr}
	${knl}
	${RuntimeObjectSystem_SRCS}
)
SOURCE_GROUP("Header Files\\External" FILES ${ext_inc})


TARGET_LINK_LIBRARIES(EagleLib
        ${OpenCV_LIBS}
        pplx
        ${Parameters_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        Qt5::OpenGL
        signals
        RuntimeObjectSystem
        RuntimeCompiler
		${LINK_LIBS}
)
if(WIN32)
    target_link_libraries(EagleLib OpenGL32)
else(WIN32)
  if(UNIX)
    target_link_libraries(EagleLib GL GLU)
  endif(UNIX)
endif(WIN32)

ADD_DEPENDENCIES(EagleLib
	parameters
	pplx
	signals
    RuntimeObjectSystem
    RuntimeCompiler
	)

install(TARGETS EagleLib
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib)

install(DIRECTORY src/ 
		DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/
		DESTINATION include
		FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_BINARY_DIR}/EagleLibConfig.cmake
		DESTINATION ./)

if(WIN32)
	install(DIRECTORY ${CMAKE_BINARY_DIR}/Debug/
		DESTINATION bin
		FILES_MATCHING PATTERN "*.pdb")
	install(DIRECTORY ${CMAKE_BINARY_DIR}/RelWithDebInfo/
		DESTINATION bin
		FILES_MATCHING PATTERN "*.pdb")
endif(WIN32)


if(UNIX)
    TARGET_LINK_LIBRARIES(EagleLib dl)
endif()
