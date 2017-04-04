PROJECT(Vclick3d)
CMAKE_POLICY(SET CMP0020 NEW)
FIND_PACKAGE(Qt5Network)
find_package(PkgConfig QUIET)
if(Qt5Network_FOUND)
	if(PkgConfig_FOUND)
		pkg_check_modules(GLOOX QUIET gloox)
		IF(GLOOX_INCLUDEDIR AND GLOOX_LIBRARIES)
			INCLUDE_DIRECTORIES(${GLOOX_INCLUDEDIR})
			add_definitions(-DHAVE_GLOOX)
		endif()
	endif(PkgConfig_FOUND)
	
	FIND_PACKAGE(CUDA REQUIRED)
	list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
	
	INCLUDE_DIRECTORIES(
  		${INCLUDE_DIRS}
  		"${CMAKE_CURRENT_SOURCE_DIR}/include"
  		${Qt5Network_INCLUDE_DIRS}
  		${Qt5Core_INCLUDE_DIRS}
	)

	find_package(PCAP QUIET)

	if(PCAP_FOUND AND PCAP_INCLUDE_DIRS)
		include_directories(${PCAP_INCLUDE_DIR})
		add_definitions(-DHAVE_PCAP)
	endif(PCAP_FOUND AND PCAP_INCLUDE_DIRS)
	
	FILE(GLOB_RECURSE hdrs "src/*.h" "src/*.hpp")
	FILE(GLOB_RECURSE src "src/*.cpp")

	QT5_WRAP_CPP(MOC ${hdrs})
	LINK_DIRECTORIES(${LINK_DIRS})
	CUDA_ADD_LIBRARY(Vclick3d SHARED ${hdrs} ${src} ${MOC})
	ADD_DEPENDENCIES(Vclick3d Aquila)

	RCC_LINK_LIB(Vclick3d
		${OpenCV_LIBS}
		Qt5::Network
		Qt5::Core
		Aquila
		${LINK_LIBS}
		${Aquila_LIBRARIES}
	)
	if(PCAP_LIBRARY)
		rcc_link_lib(Vclick3d ${PCAP_LIBRARY})
	endif()
	if(GLOOX_LIBRARIES)
		rcc_link_lib(Vclick3d ${GLOOX_LIBRARIES})
	endif()

		INCLUDE(../PluginTemplate.cmake)
endif(Qt5Network_FOUND)