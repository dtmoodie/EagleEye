###############################################################################
# Find Flann
#
# This sets the following variables:
# FLANN_FOUND - True if FLANN was found.
# FLANN_INCLUDE_DIRS - Directories containing the FLANN include files.
# FLANN_LIBRARIES - Libraries needed to use FLANN.
# FLANN_DEFINITIONS - Compiler flags for FLANN.


find_path(Aquila_FLANN_INCLUDE_DIR flann/flann.hpp
    HINTS ${PC_FLANN_INCLUDEDIR} ${PC_FLANN_INCLUDE_DIRS})

find_library(Aquila_FLANN_LIBRARY_DEBUG flann_cpp_s
    HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS})
find_library(Aquila_FLANN_LIBRARY_RELEASE flann_cpp_sd

    HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS})
	
find_library(Aquila_FLANN_CUDA_LIBRARY_RELEASE flann_cuda_sd
	HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS})
	
find_library(Aquila_FLANN_CUDA_LIBRARY_DEBUG flann_cuda_sd
	HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS})
	
set(Aquila_FLANN_INCLUDE_DIRS ${FLANN_INCLUDE_DIR})
set(Aquila_FLANN_LIBRARIES "debug;${Aquila_FLANN_LIBRARY_DEUBG};optimzied;${Aquila_FLANN_LIBRARY_RELEASE};debug;${Aquila_FLANN_CUDA_LIBRARY_DEBUG};optimized;${Aquila_FLANN_CUDA_LIBRARY_RELEASE}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Flann DEFAULT_MSG
    Aquila_FLANN_LIBRARY_DEBUG Aquila_FLANN_INCLUDE_DIR)

mark_as_advanced(Aquila_FLANN_LIBRARY_DEBUG Aquila_FLANN_LIBRARY_RELEASE Aquila_FLANN_CUDA_LIBRARY_RELEASE Aquila_FLANN_CUDA_LIBRARY_DEBUG Aquila_FLANN_LIBRARIES Aquila_FLANN_INCLUDE_DIR)