
set(mxnet_ROOT "" CACHE PATH "Root directory of mxnet install")
rcc_find_library(mxnet_LIBRARY_RELEASE mxnet
  PATHS
    ${mxnet_ROOT}/lib
    ${mxnet_ROOT}/build
    ${mxnet_ROOT}/RelWithDebInfo
    ${mxnet_ROOT}/Release
)

rcc_find_library(mxnet_LIBRARY_DEBUG mxnetd
  PATHS
    ${mxnet_ROOT}/lib
    ${mxnet_ROOT}/build
    ${mxnet_ROOT}/Debug
)

rcc_find_path(mxnet_INCLUDE mxnet/mxrtc.h
  PATHS
    ${mxnet_ROOT}/include
    ${mxnet_ROOT}/../include
)

rcc_find_path(mxnet_BIN_DIR mxnet.dll
  PATHS
    ${mxnet_ROOT}/bin
    ${mxnet_ROOT}/Debug
    ${mxnet_ROOT}/RelWithDebInfo
    ${mxnet_ROOT}/Release
)

rcc_find_path(dlpack_INCLUDE_DIR dlpack/dlpack.h
    PATHS
    ${mxnet_ROOT}/dlpack/include
)

rcc_find_path(nnvm_INCLUDE_DIR  nnvm/c_api.h
    PATHS ${mxnet_ROOT}/nnvm/include
)

rcc_find_path(mxnet_cpp_INCLUDE mxnet-cpp/MxNetCpp.h
    PATHS ${mxnet_ROOT}/cpp-package/include
          ${mxnet_ROOT}/include
)

rcc_find_path(dmlc_INCLUDE dmlc/base.h
  PATHS
    ${mxnet_ROOT}/dmlc-core/include
    ${mxnet_ROOT}/../dmlc-core/include
)

rcc_find_path(mshadow_INCLUDE mshadow/tensor.h
  PATHS
  ${mxnet_ROOT}/mshadow
  ${mxnet_ROOT}/mshadow/include
  ${mxnet_ROOT}/../mshadow
)

rcc_find_library(dmlc_CORE_LIBRARY_RELEASE dmlccore dmlc
  PATHS
    ${mxnet_ROOT}/lib
    ${mxnet_ROOT}/build/dmlc-core
    ${mxnet_ROOT}/build/dmlc-core/RelWithDebInfo
    ${mxnet_ROOT}/build/dmlc-core/Release
    ${mxnet_ROOT}/dmlc-core
)

rcc_find_library(dmlc_CORE_LIBRARY_DEBUG dmlccored dmlc
  PATHS
    ${mxnet_ROOT}/lib
    ${mxnet_ROOT}/build/dmlc-core/Debug
)

find_package(OpenBLAS QUIET)

if(OpenBLAS_FOUND)

  find_package(CUDA REQUIRED)
  set(mxnet_INCLUDE_DIRS "${dmlc_INCLUDE};${mxnet_INCLUDE};${mshadow_INCLUDE};${CUDA_TOOLKIT_INCLUDE};${mxnet_cpp_INCLUDE};${nnvm_INCLUDE_DIR};${mxnet_ROOT}/src;${dlpack_INCLUDE_DIR}")
  find_package(CUDNN QUIET)
  if(CUDNN_FOUND)
    set(mxnet_FOUND OFF)

    if(mxnet_INCLUDE AND mxnet_cpp_INCLUDE AND nnvm_INCLUDE_DIR)
      if((mxnet_LIBRARY_DEBUG OR mxnet_LIBRARY_RELEASE) AND (dmlc_CORE_LIBRARY_DEBUG OR dmlc_CORE_LIBRARY_RELEASE))
        set(mxnet_FOUND ON)
        add_library(mxnet SHARED IMPORTED)
        set_target_properties(mxnet
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                "${mxnet_cpp_INCLUDE};${OpenBLAS_INCLUDE_DIR};${CUDNN_INCLUDE_DIR};${mxnet_INCLUDE};${nnvm_INCLUDE_DIR};${dmlc_INCLUDE};${CUDA_TOOLKIT_INCLUDE}"
        )
      endif((mxnet_LIBRARY_DEBUG OR mxnet_LIBRARY_RELEASE) AND (dmlc_CORE_LIBRARY_DEBUG OR dmlc_CORE_LIBRARY_RELEASE))

      if(dmlc_CORE_LIBRARY_DEBUG OR dmlc_CORE_LIBRARY_RELEASE)
        add_library(dmlc_core STATIC IMPORTED)
        set_target_properties(dmlc_core
          PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${dmlc_INCLUDE}
        )
      endif(dmlc_CORE_LIBRARY_DEBUG OR dmlc_CORE_LIBRARY_RELEASE)

      if(dmlc_CORE_LIBRARY_DEBUG)
        set_property(TARGET dmlc_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
        set_target_properties(dmlc_core PROPERTIES
          IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "cudnn;${OpenBLAS_LIB}"
          IMPORTED_IMPLIB_DEBUG ${dmlc_CORE_LIBRARY_DEBUG}
        )
      endif(dmlc_CORE_LIBRARY_DEBUG)

      if(dmlc_CORE_LIBRARY_RELEASE)
        set_property(TARGET dmlc_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
        set_target_properties(mxnet PROPERTIES
          IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "dmlc_core;cudnn;${OpenBLAS_LIB}"
          IMPORTED_IMPLIB_RELEASE ${dmlc_CORE_LIBRARY_RELEASE}
        )
        set_property(TARGET dmlc_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
        set_target_properties(mxnet PROPERTIES
          IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "cudnn;${OpenBLAS_LIB}"
          IMPORTED_IMPLIB_RELWITHDEBINFO "${dmlc_CORE_LIBRARY_RELEASE}"
        )
      endif(dmlc_CORE_LIBRARY_RELEASE)

      if(mxnet_LIBRARY_DEBUG)
        set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
        set_target_properties(mxnet PROPERTIES
          IMPORTED_IMPLIB_DEBUG ${mxnet_LIBRARY_DEBUG}
          IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "dmlc_core;cudnn;${OpenBLAS_LIB}"
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
            IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "dmlc_core;cudnn;${OpenBLAS_LIB}"
          )

      endif(mxnet_LIBRARY_DEBUG)

      if(mxnet_LIBRARY_RELEASE)
        set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
        set_target_properties(mxnet PROPERTIES
          IMPORTED_IMPLIB_RELEASE ${mxnet_LIBRARY_RELEASE}
          IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "dmlc_core;cudnn;${OpenBLAS_LIB}"
        )

        set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
        set_target_properties(mxnet PROPERTIES
          IMPORTED_IMPLIB_RELWITHDEBINFO ${mxnet_LIBRARY_RELEASE}
          IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "dmlc_core;cudnn;${OpenBLAS_LIB}"
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
      set_target_properties(mxnet PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES "dmlc_core;cudnn;${OpenBLAS_LIB}"
      )
    endif(mxnet_INCLUDE AND mxnet_cpp_INCLUDE AND nnvm_INCLUDE_DIR)
  endif(CUDNN_FOUND)
endif(OpenBLAS_FOUND)

