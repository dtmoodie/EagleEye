project(EaglePython)

set(src "")
set(hdr "")
set(MOC "")
set(libs "")
set(includes_ "${CMAKE_CURRENT_SOURCE_DIR}/include")


# ---------------------- PythonQT section -------------------------
find_package(Qt5 QUIET COMPONENTS Core Gui Widgets)
if(Qt5_FOUND)
    find_package(PythonInterp QUIET)
    if(PythonInterp_FOUND)
        set(_python_qt_name Qt5Python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
        find_package(${_python_qt_name} QUIET)
        if(${_python_qt_name}_FOUND)
            include_directories("pythonqt/include")
            include_directories("${${_python_qt_name}_DIR}/../../../include")
            file(GLOB_RECURSE __src "pythonqt/src/*.cpp")
            file(GLOB_RECURSE __hdr "pythonqt/include/*.h" "pythonqt/include/*.hpp")
            QT5_WRAP_CPP(MOC ${__hdr})
            list(APPEND includes_ "${CMAKE_CURRENT_SOURCE_DIR}/pythonqt/include")
            list(APPEND src "${__src}")
            list(APPEND hdr "${__hdr}")
            list(APPEND libs "${_python_qt_name}")
            message("Building PythonQt")
        endif()
    endif()
endif()

# ----------------------- Boost::Python section ---------------------
unset(PYTHON_LIBRARY)
find_package(PythonLibs 2.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )
find_package(Boost REQUIRED COMPONENTS Python)
include_directories( ${Boost_INCLUDE_DIR} )
file(GLOB_RECURSE _src "src/*.cpp")
file(GLOB_RECURSE _hdr "include/*.h" "include/*.hpp")
list(APPEND src "${_src}")
list(APPEND hdr "${_hdr}")

# ----------------------- Build library ---------------------
include_directories(${EagleLib_INCLUDE_DIRS} ${INCLUDE_DIRS})
add_library(EaglePython SHARED ${hdr} ${src} ${MOC})
link_directories("${CMAKE_BINARY_DIR}")
target_link_libraries(EaglePython 
    ${libs} 
    ${Boost_LIBRARIES} 
    EagleLib
    RuntimeObjectSystem
    RuntimeCompiler
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cupti_LIBRARY}
    ${CUDA_nppi_LIBRARY}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_npps_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${PYTHON_LIBRARY})
message(${includes_})
set(EaglePython_INCLUDE "${includes_}" CACHE PATH "" FORCE)