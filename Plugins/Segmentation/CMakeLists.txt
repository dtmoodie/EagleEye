
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
CMAKE_POLICY(SET CMP0020 NEW)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(CUDA REQUIRED)

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec cudastereo)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

INCLUDE_DIRECTORIES(
        ${EagleLib_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)
LINK_DIRECTORIES(${LINK_DIRS})
file(GLOB_RECURSE src "*.cpp")
file(GLOB_RECURSE hdr "*.h" "*.hpp" "*.cuh")
file(GLOB_RECURSE knl "*.cu")

#SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")

cuda_add_library(Segmentation SHARED ${src} ${hdr})
add_dependencies(Segmentation EagleLib)
target_link_libraries(Segmentation
    EagleLib
    ${OpenCV_LIBS}
	
)

#set_target_properties(Segmentation PROPERTIES
    #RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    #ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    #LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH})
IF(TARGET fastms)
    add_dependencies(Segmentation fastms)
    target_link_libraries(Segmentation fastms ${fastms_LIBS})
    add_definitions(-DFASTMS_FOUND)
    add_definitions(-DFASTMS_INCLUDE=${fastms_INCLUDE_DIRECTORIES})
    INCLUDE_DIRECTORIES(${fastms_INCLUDE_DIRECTORIES})
	message(STATUS "FastMS include: ${fastms_INCLUDE_DIRECTORIES}") 

ENDIF()
