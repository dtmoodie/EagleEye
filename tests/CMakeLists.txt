cmake_minimum_required(VERSION 2.8)

project(Tests)
IF(${CMAKE_VERSION} VERSION_GREATER 3.1)
	CMAKE_POLICY(SET CMP0053 OLD)
ENDIF()

cmake_policy(SET CMP0020 NEW)

set(BOOST_REQUIRED_MODULES program_options)
find_package(Boost 1.47.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})

find_package(OpenCV 3.0 REQUIRED core imgproc highgui cudaimgproc cudawarping cudafeatures2d cudaoptflow cudacodec)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(
	${EagleLib_INCLUDE_DIRECTORIES}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${PPLX_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${LINK_DIRS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

FILE(GLOB CPP_TESTS "*.cpp")

FOREACH( test ${CPP_TESTS} )
  get_filename_component( name ${test} NAME_WE )
  add_executable(${name} ${test})
  add_dependencies(${name} EagleLib)
  target_link_libraries(${name} ${LINK_LIBS} ${Boost_LIBRARIES})
ENDFOREACH( test CPP_TESTS )
#add_executable(test_shared_ptr test_shared_ptr.cpp)
#add_dependencies(test_shared_ptr EagleLib)
#target_link_libraries(test_shared_ptr EagleLib ${LINK_LIBS} ${Boost_LIBRARIES})

#add_executable(test_tree test_tree.cpp)
#add_dependencies(test_tree EagleLib)
#target_link_libraries(test_tree EagleLib ${LINK_LIBS} ${Boost_LIBRARIES})

#add_executable(test_rcc test_rcc.cpp)
#add_dependencies(test_rcc EagleLib)
#target_link_libraries(test_rcc EagleLib ${LINK_LIBS} ${Boost_LIBRARIES})

#add_executable(perf_test perf_test.cpp)
#add_dependencies(perf_test EagleLib)
#target_link_libraries(perf_test EagleLib ${LINK_LIBS} ${Boost_LIBRARIES})

#add_executable(bind_test bind_test.cpp)
#target_link_libraries(perf_test EagleLib ${LINK_LIBS} ${Boost_LIBRARIES})

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
cuda_add_executable(test_scoped_buffer test_scoped_buffer.cu)
target_link_libraries(test_scoped_buffer  EagleLib ${LINK_LIBS} ${Boost_LIBRARIES})

cuda_add_executable(test_pplx test_pplx.cu)
target_link_libraries(test_pplx EagleLib ${LINK_LIBS} ${PPLX_LIBRARIES})