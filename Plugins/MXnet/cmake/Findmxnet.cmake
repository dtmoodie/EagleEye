
set(mxnet_ROOT "" CACHE PATH "Root directory of mxnet install")
rcc_find_library(mxnet_LIBRARY_RELEASE mxnet PATHS ${mxnet_ROOT}/lib)
rcc_find_library(mxnet_LIBRARY_DEBUG mxnetd PATHS ${mxnet_ROOT}/lib)

rcc_find_path(mxnet_INCLUDE mxnet/mxrtc.h PATHS ${mxnet_ROOT}/include)
rcc_find_path(mxnet_BIN_DIR mxnet.dll PATHS ${mxnet_ROOT}/bin)

rcc_find_library(DMLC_CORE_LIBRARY_RELEASE dmlccore PATHS ${mxnet_ROOT}/lib)
rcc_find_library(DMLC_CORE_LIBRARY_DEBUG dmlccored PATHS ${mxnet_ROOT}/lib)

find_package(OpenBLAS QUIET)
  include_directories(${OpenBLAS_INCLUDE_DIR})
if(OpenBLAS_FOUND)
find_package(CUDA REQUIRED)
  include_directories(${CUDA_TOOLKIT_INCLUDE})

find_package(CUDNN QUIET)
if(CUDNN_FOUND)
  include_directories(${CUDNN_INCLUDE_DIR})
  
	set(mxnet_FOUND OFF)
		if(mxnet_INCLUDE)
		  if(mxnet_LIBRARY_DEBUG OR mxnet_LIBRARY_RELEASE)
			set(mxnet_FOUND ON)
			add_library(mxnet SHARED IMPORTED)
		  endif(mxnet_LIBRARY_DEBUG OR mxnet_LIBRARY_RELEASE)

		  if(DMLC_CORE_LIBRARY_DEBUG OR DMLC_CORE_LIBRARY_RELEASE)
			add_library(dmlc_core STATIC IMPORTED)
		  endif(DMLC_CORE_LIBRARY_DEBUG OR DMLC_CORE_LIBRARY_RELEASE)

		  if(DMLC_CORE_LIBRARY_DEBUG)
			set_property(TARGET dmlc_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
				set_target_properties(dmlc_core PROPERTIES
				  IMPORTED_IMPLIB_DEBUG "cudnn;${OpenBLAS_LIB}"
				  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG ""
			)
		  endif(DMLC_CORE_LIBRARY_DEBUG)

		  if(DMLC_CORE_LIBRARY_RELEASE)
				set_property(TARGET dmlc_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
				set_target_properties(mxnet PROPERTIES
				  IMPORTED_IMPLIB_DEBUG "cudnn;${OpenBLAS_LIB}"
				  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE ""
			)
				set_property(TARGET dmlc_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
				set_target_properties(mxnet PROPERTIES
				  IMPORTED_IMPLIB_DEBUG "dmlc_core;cudnn;${OpenBLAS_LIB}"
				  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO ""
			)
		  endif(DMLC_CORE_LIBRARY_RELEASE)

		  if(mxnet_LIBRARY_DEBUG AND DMLC_CORE_LIBRARY_DEBUG)
			set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
			set_target_properties(mxnet PROPERTIES
				  IMPORTED_IMPLIB_DEBUG "${mxnet_LIBRARY_DEBUG}"
				  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "dmlc_core;cudnn;${OpenBLAS_LIB}"
				  IMPORTED_LOCATION_DEBUG "${mxnet_BIN_DIR}/mxnetd.dll"
			)
		  endif(mxnet_LIBRARY_DEBUG AND DMLC_CORE_LIBRARY_DEBUG)

		  if(mxnet_LIBRARY_RELEASE AND DMLC_CORE_LIBRARY_RELEASE)
			set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
			set_target_properties(mxnet PROPERTIES
				  IMPORTED_IMPLIB_RELEASE "${mxnet_LIBRARY_RELEASE}"
				  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "dmlc_core;cudnn;${OpenBLAS_LIB}"
				  IMPORTED_LOCATION_RELEASE "${mxnet_BIN_DIR}/mxnet.dll"
			)

			set_property(TARGET mxnet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
			set_target_properties(mxnet PROPERTIES
				  IMPORTED_IMPLIB_RELWITHDEBINFO "${mxnet_LIBRARY_RELEASE}"
				  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "dmlc_core;cudnn;${OpenBLAS_LIB}"
				  IMPORTED_LOCATION_RELWITHDEBINFO "${mxnet_BIN_DIR}/mxnet.dll"
			)
		  endif(mxnet_LIBRARY_DEBUG AND DMLC_CORE_LIBRARY_DEBUG)
		endif(mxnet_INCLUDE)
	endif(CUDNN_FOUND)
endif(OpenBLAS_FOUND)

