
set(mxnet_DIR "" CACHE PATH "Root directory of mxnet install")
find_library(mxnet_LIBRARY_RELEASE mxnet
  PATHS
    ${mxnet_DIR}/lib
    ${mxnet_DIR}/build
    ${mxnet_DIR}/RelWithDebInfo
    ${mxnet_DIR}/Release
)

find_library(mxnet_LIBRARY_DEBUG mxnetd
  PATHS
    ${mxnet_DIR}/lib
    ${mxnet_DIR}/build
    ${mxnet_DIR}/Debug
)

find_path(mxnet_INCLUDE mxnet/c_api.h
  PATHS
    ${mxnet_DIR}/include
    ${mxnet_DIR}/../include
)

find_path(mxnet_BIN_DIR mxnet.dll
  PATHS
    ${mxnet_DIR}/bin
    ${mxnet_DIR}/Debug
    ${mxnet_DIR}/RelWithDebInfo
    ${mxnet_DIR}/Release
)

find_path(dlpack_INCLUDE_DIR dlpack/dlpack.h
    PATHS
    ${mxnet_DIR}/dlpack/include
    ${mxnet_DIR}/3rdparty/dlpack/include
)

find_path(mxnet_cpp_INCLUDE mxnet-cpp/MxNetCpp.h
    PATHS ${mxnet_DIR}/cpp-package/include
          ${mxnet_DIR}/include
)

find_path(nnvm_INCLUDE nnvm/c_api.h
    PATHS
        ${mxnet_DIR}/3rdparty/nnvm/include
)

find_path(dmlc_INCLUDE dmlc/base.h
  PATHS
    ${mxnet_DIR}/dmlc-core/include
    ${mxnet_DIR}/../dmlc-core/include
    ${mxnet_DIR}/3rdparty/dmlc-core/include
)

find_path(mshadow_INCLUDE mshadow/tensor.h
  PATHS
  ${mxnet_DIR}/mshadow
  ${mxnet_DIR}/mshadow/include
  ${mxnet_DIR}/../mshadow
  ${mxnet_DIR}/3rdparty/mshadow
)


find_package(OpenBLAS QUIET)
if(NOT OpenBLAS_FOUND)
    set(mxnet_FOUND OFF)
    return()
endif(NOT OpenBLAS_FOUND)

find_package(CUDA QUIET)
if(NOT CUDA_FOUND)
    set(mxnet_FOUND OFF)
    return()
endif(NOT CUDA_FOUND)

find_package(CUDNN QUIET)
if(NOT CUDNN_FOUND)
    set(mxnet_FOUND OFF)
    return()
endif(NOT CUDNN_FOUND)

if(NOT (nnvm_INCLUDE AND dmlc_INCLUDE AND dlpack_INCLUDE_DIR AND mshadow_INCLUDE AND mxnet_INCLUDE AND mxnet_cpp_INCLUDE AND (mxnet_LIBRARY_RELEASE OR mxnet_LIBRARY_DEBUG)))
    set(mxnet_FOUND OFF)
    return()
endif(NOT (nnvm_INCLUDE AND dmlc_INCLUDE AND dlpack_INCLUDE_DIR AND mshadow_INCLUDE AND mxnet_INCLUDE AND mxnet_cpp_INCLUDE AND (mxnet_LIBRARY_RELEASE OR mxnet_LIBRARY_DEBUG)))

set(mxnet_INCLUDE_DIRS "${mxnet_INCLUDE};${mxnet_cpp_INCLUDE};${dmlc_INCLUDE};${mshadow_INCLUDE};${dlpack_INCLUDE_DIR};${nnvm_INCLUDE}")

set(mxnet_FOUND ON)
add_library(mxnet SHARED IMPORTED)
set_target_properties(mxnet
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
        "${mxnet_INCLUDE_DIRS}"
)

set_target_properties(mxnet PROPERTIES
    IMPORTED_LINK_INTERFACE_LIBRARIES "cudnn;${OpenBLAS_LIB}"
)


if(mxnet_LIBRARY_DEBUG)
    set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(mxnet PROPERTIES
        IMPORTED_IMPLIB_DEBUG ${mxnet_LIBRARY_DEBUG}
        IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "cudnn;${OpenBLAS_LIB}"
    )
    if(MSVC)
        set_target_properties(mxnet PROPERTIES
            IMPORTED_LOCATION_DEBUG "${mxnet_BIN_DIR}/mxnetd.dll"
        )
    endif(MSVC)

elseif(mxnet_LIBRARY_RELEASE)
    # if debug not found, link to release build in debug config
    message(STATUS "Populating mxnet DEBUG config with RELEASE library")
    set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(mxnet PROPERTIES
        IMPORTED_IMPLIB_DEBUG ${mxnet_LIBRARY_RELEASE}
        IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "cudnn;${OpenBLAS_LIB}"
    )
endif(mxnet_LIBRARY_DEBUG)

if(mxnet_LIBRARY_RELEASE)
    set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(mxnet PROPERTIES
        IMPORTED_IMPLIB_RELEASE ${mxnet_LIBRARY_RELEASE}
        IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "cudnn;${OpenBLAS_LIB}"
    )

    set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
    set_target_properties(mxnet PROPERTIES
        IMPORTED_IMPLIB_RELWITHDEBINFO ${mxnet_LIBRARY_RELEASE}
        IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "cudnn;${OpenBLAS_LIB}"
    )

    if(MSVC)
        set_target_properties(mxnet PROPERTIES
            IMPORTED_LOCATION_RELWITHDEBINFO "${mxnet_BIN_DIR}/mxnet.dll"
        )
        set_target_properties(mxnet PROPERTIES
            IMPORTED_LOCATION_RELEASE "${mxnet_BIN_DIR}/mxnet.dll"
        )
    endif(MSVC)
endif(mxnet_LIBRARY_RELEASE)


