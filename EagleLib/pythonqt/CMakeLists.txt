project(EagleLib_pythonQT)

find_package(Qt5Python35)
include_directories("${Qt5Python35_DIR}/../../../include")
file(GLOB_RECURSE src "src/*.cpp" "src/*.h" "src/*.hpp")

include_directories(
	"src"
	${EagleLib_INCLUDE_DIRS}
	)
  
add_library(EagleLib_pythonQT SHARED ${src})
target_link_libraries(EagleLib_pythonQT Qt5Python35)
