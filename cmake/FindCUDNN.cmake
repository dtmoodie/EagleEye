rcc_find_library(CUDNN_LIBRARY cudnn.lib cudnn.so libcudnn.so cudnn HINTS "${CUDNN_DIR}/lib/x64/" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
rcc_find_path(CUDNN_LIB_DIR cudnn.lib cudnn.so libcudnn.so cudnn HINTS "${CUDNN_DIR}/lib/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
rcc_find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${CUDNN_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/include")
rcc_find_path(CUDNN_BIN_DIR cudnn64_4.dll cudnn64_3.dll libcudnn.so cudnn cudnn.so HINTS "${CUDNN_DIR}/bin" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

if(CUDNN_LIBRARY)
  add_library(cudnn SHARED IMPORTED)
  set_property(TARGET cudnn APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
  set_target_properties(cudnn PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${CUDNN_LIBRARY}"
	IMPORTED_LOCATION_DEBUG "${CUDNN_BIN_DIR}/cudnn64_5.dll"
  )

  set_property(TARGET cudnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
  set_target_properties(cudnn PROPERTIES
    IMPORTED_IMPLIB_RELEASE "${CUDNN_LIBRARY}"
	IMPORTED_LOCATION_RELEASE "${CUDNN_BIN_DIR}/cudnn64_5.dll"
  )
  
  set_property(TARGET cudnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
  set_target_properties(cudnn PROPERTIES
    IMPORTED_IMPLIB_RELWITHDEBINFO "${CUDNN_LIBRARY}"
	IMPORTED_LOCATION_RELWITHDEBINFO "${CUDNN_BIN_DIR}/cudnn64_5.dll"
  )
  set(CUDNN_FOUND ON)
else()
  set(CUDNN_FOUND OFF)
endif()
