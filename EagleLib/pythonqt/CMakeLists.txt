project(EaglePython)

find_package(Qt5Python35)

if(Qt5Python35_FOUND)
  message(STATUS "Building against Qt5Python35")
  include_directories("${Qt5Python35_DIR}/../../../include")
  file(GLOB_RECURSE src "src/*.cpp" "src/*.h")
  file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp")
  include_directories(
	  "include"
	  "src"
	  ${EagleLib_INCLUDE_DIRS}
	  ${INCLUDE_DIRS}
	)
	LINK_DIRECTORIES(${LINK_DIRS})
  
  add_library(EaglePython SHARED ${src} ${hdr})
  target_link_libraries(EaglePython Qt5Python35 EagleLib)
  set(EaglePython_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include CACHE PATH "")
else()
  message("Could not find Qt5Python35")
endif()