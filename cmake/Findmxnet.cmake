
set(MXNET_DIR "" CACHE PATH "")

find_library(MXNET_LIBRARY libmxnet
    PATHS ${MXNET_DIR}/lib
	)
	
find_path(MXNET_INCLUDE_DIRS mxnet/mxrtc.h
    PATHS ${MXNET_DIR}/include
	)
set(MXNET_FOUND FALSE)	
if(MXNET_INCLUDE_DIRS AND MXNET_LIBRARY)
  message("FOUND MXNET")
  set(MXNET_FOUND TRUE)
endif()
	