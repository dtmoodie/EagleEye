project(EagleLib)
CMAKE_POLICY(SET CMP0020 NEW)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(OPENCV_STATIC OFF)
find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudabgsegm)
find_package(CUDA REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets)
cmake_policy(SET CMP0012 NEW)

IF(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib)
ELSE(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib64/cudadevrt)
ENDIF(WIN32)
ADD_DEFINITIONS(-DCVAPI_EXPORTS)
ADD_DEFINITIONS(-DOPENCV_FOUND)
ADD_DEFINITIONS(-DEagleLIB_EXPORTS)
ADD_DEFINITIONS(${Parameters_DEFS})
LIST(APPEND defs_ -DCVAPI_EXPORTS)
LIST(APPEND defs_ -DOPENCV_FOUND)
LIST(APPEND defs_ -DEagleLIB_EXPORTS)
LIST(APPEND defs_ ${Parameters_DEFS})
IF(EAGLE_LIB_JETSON)
LIST(APPEND defs_ "-DJETSON")
ENDIF()
IF(NOT WIN32)
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono regex log log_setup)
ENDIF(NOT WIN32)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES}) 

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE ext_inc "include/external_includes")
ADD_SUBDIRECTORY("parameters")



INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
	${Parameters_INCLUDE}
	${Qt5Gui_INCLUDE_DIRS} 
	${Qt5Widgets_INCLUDE_DIRS}
	)



# for RCC to be able to recompile modules
LIST(GET OpenCV_INCLUDE_DIRS 0 OpenCV_INCLUDE_DIR)
LIST(GET OpenCV_LIB_DIR 0 CV_LIB_DIR)
if(${CV_LIB_DIR} MATCHES NOTFOUND)
 message("OpenCV_LIB_DIR not set, attempting to find library dir")
endif()

SET(IObject_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

SET(RCC_INCLUDE_DEPENDENCIES "${Boost_INCLUDE_DIRS};${OpenCV_INCLUDE_DIR};${CUDA_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/src;${Parameters_INCLUDE}")

IF(WIN32)
	SET(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIRS};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib/x64;${Parameters_LIB_DIRS_RELEASE};${CMAKE_CURRENT_BINARY_DIR}/../Debug")
	SET(RCC_LIBRARY_DIRS_RELEASE "${Boost_LIBRARY_DIRS};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib/x64;${Parameters_LIB_DIRS_RELEASE};${CMAKE_CURRENT_BINARY_DIR}/../Release;")
	SET(RCC_LIBRARY_DIRS_RELWITHDEBINFO "${Boost_LIBRARY_DIRS};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib/x64;${Parameters_LIB_DIRS_RELEASE};${CMAKE_CURRENT_BINARY_DIR}/../RelWithDebInfo")
ELSE()
        SET(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIRS};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64;${LIBRARY_OUTPUT_PATH};${Parameters_LIB_DIRS_DEBUG}")
	SET(RCC_LIBRARY_DIRS_RELEASE "${Boost_LIBRARY_DIRS};${CV_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64;${LIBRARY_OUTPUT_PATH};${Parameters_LIB_DIRS_RELEASE}")
ENDIF()

ADD_SUBDIRECTORY("RCC")

INCLUDE_DIRECTORIES(${RCC_INCLUDE_DIRS})



IF(NOT WIN32)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
ENDIF()


#
#				Export variables for other projects to use
#
set(EagleLib_INCLUDE_DIRECTORIES "${Parameters_INCLUDE};${CMAKE_CURRENT_SOURCE_DIR}/src;${Boost_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS};${RCC_INCLUDE_DIRS}" PARENT_SCOPE)
set(EagleLib_LIBRARIES "EagleLib;${Boost_LIBRARIES};${OpenCV_LIBS};${Parameters_LIBRARIES}" PARENT_SCOPE)
SET(EagleLib_LIB_DIRS_DEBUG "${Parameters_LIB_DIRS_DEBUG}" PARENT_SCOPE)
SET(EagleLib_LIB_DIRS_RELEASE "${Parameters_LIB_DIRS_RELEASE}" PARENT_SCOPE)
SET(EagleLib_DEFS "${defs_}" PARENT_SCOPE)

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
        RuntimeCompiler
        RuntimeObjectSystem
        libParameter
        ${Parameters_LIBRARIES}
)
ADD_DEPENDENCIES(EagleLib
	RuntimeCompiler
	RuntimeObjectSystem
	libParameter
	)


if(UNIX)
    TARGET_LINK_LIBRARIES(EagleLib dl)
endif()
