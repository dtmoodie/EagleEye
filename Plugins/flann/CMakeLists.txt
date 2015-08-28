CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)

FIND_PACKAGE(FLANN REQUIRED)

IF(FLANN_FOUND)
	FILE(GLOB_RECURSE src "src/*.cpp")
	FILE(GLOB_RECURSE hdr "src/*.hpp" "src/*.h")
	FILE(GLOB_RECURSE knl "src/*.cu")
	
	INCLUDE_DIRECTORIES(
		${CMAKE_CURRENT_SOURCE_DIR}
		${EagleLib_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
		${FLANN_INCLUDE_DIR}
		)
		
		
	CUDA_ADD_LIBRARY(Flann ${src} ${hdr} ${knl})
	add_dependencies(Flann EagleLib)
	target_link_libraries(Flann
        EagleLib
        ${OpenCV_LIBS}
        ${CUDA_CUBLAS_LIBRARIES}
        ${LINK_LIBS}
		${FLANN_LIBRARY}
		)


ELSE(FLANN_FOUND)

MESSAGE("Flann not found, not building flann module")
ENDIF(FLANN_FOUND)