
find_package(Wt QUIET)
if(Wt_FOUND)
  find_package(CUDA REQUIRED)
  include_directories(${Wt_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  file(GLOB_RECURSE src "*.cpp" "*.hpp")
  add_executable(WebUi ${src})
  target_link_libraries(WebUi 
    Aquila 
    MetaObject 
    MetaParameters
    debug;${Wt_HTTP_DEBUG_LIBRARY}
    debug;${Wt_DEBUG_LIBRARY}
    optimized;${Wt_LIBRARY}
    optimized;${Wt_HTTP_LIBRARY}
  )
else(Wt_FOUND)
  if(RCC_VERBOSE_CONFIG)
    message(STATUS "Wt not found, not building web interface")
  endif(RCC_VERBOSE_CONFIG)
endif(Wt_FOUND)
