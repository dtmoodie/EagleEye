PROJECT(Vclick3d)
CMAKE_POLICY(SET CMP0020 NEW)
FIND_PACKAGE(Qt5Network)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  	pkg_check_modules(GLOOX QUIET gloox)
  	
	FIND_PACKAGE(CUDA REQUIRED)
	list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
	INCLUDE_DIRECTORIES(
  		${INCLUDE_DIRS}
  		"${CMAKE_CURRENT_SOURCE_DIR}/include"
  		${Qt5Network_INCLUDE_DIRS}
  		${Qt5Core_INCLUDE_DIRS}
	)

	find_package(PCAP QUIET)

	if(PCAP_FOUND)
		include_directories(${PCAP_INCLUDE_DIR})
		add_definitions(-DHAVE_PCAP)
	endif(PCAP_FOUND)
	IF(GLOOX_INCLUDEDIR AND GLOOX_LIBRARIES)
		INCLUDE_DIRECTORIES(${GLOOX_INCLUDEDIR})
		add_definitions(-DHAVE_GLOOX)
	endif()

	FILE(GLOB_RECURSE hdrs "src/*.h" "src/*.hpp")
	FILE(GLOB_RECURSE src "src/*.cpp")

	QT5_WRAP_CPP(MOC ${hdrs})
	LINK_DIRECTORIES(${LINK_DIRS})
	CUDA_ADD_LIBRARY(Vclick3d SHARED ${hdrs} ${src} ${MOC})
	ADD_DEPENDENCIES(Vclick3d EagleLib)


	RCC_LINK_LIB(Vclick3d
		${OpenCV_LIBS}
		Qt5::Network
		Qt5::Core
		EagleLib
		${LINK_LIBS} 
		${GLOOX_LIBRARIES}
		${EagleLib_LIBRARIES}
		${PCAP_LIBRARY}
	)

		INCLUDE(../PluginTemplate.cmake)



endif(PkgConfig_FOUND)
