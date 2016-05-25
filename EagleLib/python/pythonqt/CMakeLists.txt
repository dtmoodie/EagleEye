project(EaglePython)
unset(PYTHON_LIBRARY)
find_package(Qt5Python35)

if(Qt5_FOUND)
    find_package(Qt5 QUIET COMPONENTS Core Gui Widgets)
    
    if(Qt5Python35_FOUND)
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

      QT5_WRAP_CPP(MOC ${hdr})

      add_library(EaglePython SHARED ${src} ${hdr} ${MOC})
      target_link_libraries(EaglePython Qt5Python35 EagleLib)
      set(EaglePython_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include CACHE PATH "")
    else()
      message("Could not find Qt5Python35")
    endif()
endif()
