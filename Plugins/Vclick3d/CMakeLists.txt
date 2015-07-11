PROJECT(Vclick3d)


FILE(GLOB_RECURSE hdrs "include/*.h")
FILE(GLOB_RECURSE src "src/*.cpp")




ADD_LIBRARY(Vclick3d SHARED ${hdrs} ${src})