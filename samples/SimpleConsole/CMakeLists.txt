project(samples)
CMAKE_POLICY(SET CMP0020 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")

if(NOT TARGET aquila_core)
    find_package(Aquila REQUIRED COMPONENTS core)
endif()

find_package(CUDA REQUIRED)

file(GLOB src "*.cpp")
file(GLOB hdr "*.h" "*.hpp")

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(GraphExecutor ${src} ${hdr})

target_link_libraries(GraphExecutor
	${Boost_LIBRARIES}
    metaobject_core
    aquila_core
    RuntimeObjectSystem
    RuntimeCompiler
    metaobject_metaparams
    aquila_types
)

if(WITH_PYTHON AND MO_HAVE_PYTHON)
	target_link_libraries(GraphExecutor ${PYTHON_LIBRARY})
endif(WITH_PYTHON AND MO_HAVE_PYTHON)

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


