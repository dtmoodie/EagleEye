cmake_minimum_required(VERSION 3.2)

project(NodeEditor)

option(BUILD_EXAMPLES "Build Examples" ON)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Find the QtWidgets library
find_package(Qt5 COMPONENTS
             Core
             Widgets
             Gui
             OpenGL)

add_definitions(${Qt5Widgets_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

IF (MSVC)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF (MSVC)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#############################################################

file(GLOB_RECURSE LIB_CPPS  ./src/*.cpp )

set(NodeEditor_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src" CACHE PATH "" FORCE)

qt5_add_resources(RESOURCES ./resources/resources.qrc)

# Tell CMake to create the helloworld executable
add_library(NodeEditor SHARED ${LIB_CPPS} ${RESOURCES})

target_include_directories(NodeEditor INTERFACE "include")

target_compile_definitions(NodeEditor PUBLIC "-DNODE_EDITOR_SHARED")
target_compile_definitions(NodeEditor PRIVATE "-DNODE_EDITOR_EXPORTS")

target_link_libraries(NodeEditor
                      Qt5::Core
                      Qt5::Widgets
                      Qt5::Gui
                      Qt5::OpenGL)

install(TARGETS NodeEditor
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
      DESTINATION include/NodeEditor
      FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY src/
      DESTINATION include/NodeEditor
      FILES_MATCHING PATTERN "*.hpp"
)


if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
