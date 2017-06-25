project(samples)
CMAKE_POLICY(SET CMP0020 NEW)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
    CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")

if(NOT TARGET aquila_core)
    find_package(Aquila REQUIRED COMPONENTS core)
endif()
find_package(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})

LINK_DIRECTORIES(${LINK_DIRS})

find_package(CUDA REQUIRED)
if(WITH_WT)
	find_package(Wt QUIET)
	if(Wt_FOUND)
		include_directories(${Wt_INCLUDE_DIR})
		add_definitions(-DHAVE_WT)
	if(WIN32)
		SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
		find_library(Wt_bin NAMES wt PATHS ${Wt_DIR} PATH PATH_SUFFIXES bin)
		get_filename_component(Wt_bin_dir "${Wt_bin}" DIRECTORY)
		set(Wt_BIN_DIR_ ${Wt_bin_dir})
	endif(WIN32)
		set(Wt_BIN_DIR ${Wt_BIN_DIR_} CACHE PATH "" FORCE)
		set(Wt_BIN_DIR_DBG ${Wt_BIN_DIR_} CACHE PATH "" FORCE)
		set(Wt_BIN_DIR_OPT ${Wt_BIN_DIR_} CACHE PATH "" FORCE)
		set(BIN_DIRS "${BIN_DIRS};Wt" CACHE STRING "" FORCE)
	endif()
endif(WITH_WT)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
file(GLOB src "*.cpp")
file(GLOB hdr "*.h" "*.hpp")

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(GraphExecutor ${src} ${hdr})

target_link_libraries(GraphExecutor
    metaobject_core
    aquila_core
    ${Boost_LIBRARIES}
    RuntimeObjectSystem
    RuntimeCompiler
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cupti_LIBRARY}
    ${CUDA_nppi_LIBRARY}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_npps_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    metaobject_metaparams
    aquila_types
    aquila_metatypes
)
if(MetaObject_WITH_WT AND Wt_FOUND)
    target_compile_definitions(GraphExecutor
        PRIVATE "-DHAVE_WT"
    )
    target_include_directories(GraphExecutor
        PRIVATE ${Wt_INCLUDE_DIR}
    )
endif()

if(WITH_PYTHON AND HAVE_PYTHON)
	target_link_libraries(GraphExecutor ${PYTHON_LIBRARY})
endif(WITH_PYTHON AND HAVE_PYTHON)

if(Wt_FOUND AND WITH_WT)
    target_link_libraries(GraphExecutor
        debug;${Wt_HTTP_DEBUG_LIBRARY}
        debug;${Wt_DEBUG_LIBRARY}
        optimized;${Wt_LIBRARY}
        optimized;${Wt_HTTP_LIBRARY}
    )
endif(Wt_FOUND AND WITH_WT)

INSTALL(TARGETS GraphExecutor
        RUNTIME DESTINATION bin
)

set_property(TARGET GraphExecutor PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

IF(WIN32)
	GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_DEBUG)
	GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_RELEASE)
	GENERATE_WIN_DLL_PATHS(PROJECT_BIN_DIRS_RELWITHDEBINFO)

	list(APPEND PROJECT_BIN_DIRS_DEBUG ${Aquila_BIN_DIRS})
	if(RCC_VERBOSE_CONFIG)
		message(STATUS "GraphExecutor binary runtime directories:\n ${PROJECT_BIN_DIRS_DEBUG}")
	endif()
	CONFIGURE_FILE("../../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/samples/SimpleConsole/GraphExecutor.vcxproj.user @ONLY)
endif(WIN32)


