CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
project(freenect)

#set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_package(CUDA REQUIRED)

FIND_PACKAGE(libusb-1.0)

find_package(libfreenect)

IF(LIBUSB_1_FOUND AND LIBFREENECT_FOUND)
    IF(WIN32)
        set(THREADS_USE_PTHREADS_WIN32 true)
        find_package(Threads REQUIRED)
        INCLUDE_DIRECTORIES(${THREADS_PTHREADS_INCLUDE_DIR})
        SET(PTHREADS_INCLUDE_DIR "" CACHE PATH "${THREADS_PTHREADS_INCLUDE_DIR}")
        SET(PTHREADS_LIBRARY "" CACHE FILE "${PTHREADS_INCLUDE_DIR}/../lib/x64/pthreadVC2.lib")
        INCLUDE_DIRECTORIES(${PTHREADS_INCLUDE_DIR})

    ELSE(WIN32)
        LINK_DIRECTORIES("/usr/local/lib")
    ENDIF(WIN32)

    ADD_DEFINITIONS(${DEFS})
    LINK_DIRECTORIES(${LINK_DIRS})

    INCLUDE_DIRECTORIES(
        ${EagleLib_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${LIBUSB_1_INCLUDE_DIR}
        /usr/include/libusb-1.0
        ${FREENECT_INCLUDE_DIRS}
        ${LIBFREENECT_INCLUDE_DIR}
    )

    file(GLOB_RECURSE src "*.cpp")
    file(GLOB_RECURSE knl "*.cu")
    file(GLOB_RECURSE hdr "*.h" "*.hpp" "*.cuh")

    IF(WIN32)

    ELSE()
        #SET(CUDA_NVCC_FLAGS "-ccbin g++;-O2;-std=c++11;${CUDA_NVCC_FLAGS}")
    ENDIF()

    cuda_add_library(freenect SHARED ${src} ${hdr} ${knl})
    add_dependencies(freenect EagleLib)
    RCC_LINK_LIB(freenect
        EagleLib
        ${LIBFREENECT_LIBRARY}
        ${OpenCV_LIBS}
        ${CUDA_CUBLAS_LIBRARIES}
        ${LINK_LIBS}
        ${LIBUSB_1_LIBRARY}
    )
    IF(WIN32)
        TARGET_LINK_LIBRARIES(freenect ${PTHREADS_LIBRARY})
    ENDIF(WIN32)
    INCLUDE(../PluginTemplate.cmake)

ELSE(LIBUSB_1_FOUND AND LIBFREENECT_FOUND)
    message("LIBUSB not found, not building freenect lib")
ENDIF(LIBUSB_1_FOUND AND LIBFREENECT_FOUND)
