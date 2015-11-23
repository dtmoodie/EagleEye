#pragma once
#include <boost/preprocessor.hpp>

#define STRINGIFY_1(ARG1) #ARG1
#define STRINGIFY_2(ARG1, ARG2)	#ARG1, #ARG2
#define STRINGIFY_3(ARG1, ARG2, ARG3) #ARG1, #ARG2, #ARG3
#define STRINGIFY_4(ARG1, ARG2, ARG3, ARG4) #ARG1, #ARG2, #ARG3, #ARG4
#define STRINGIFY_5(ARG1, ARG2, ARG3, ARG4, ARG5) #ARG1, #ARG2, #ARG3, #ARG4, #ARG5
#ifdef _MSC_VER
#define STRINGIFY(...) 	BOOST_PP_CAT( BOOST_PP_OVERLOAD( STRINGIFY_, __VA_ARGS__ )(__VA_ARGS__), BOOST_PP_EMPTY() )
#else
#define STRINGIFY(...) 	BOOST_PP_OVERLOAD( STRINGIFY_, __VA_ARGS__ )(__VA_ARGS__)
#endif


#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define EAGLE_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define EAGLE_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define EAGLE_EXPORTS
#endif


// *************** SETUP_PROJECT_DEF ********************
#ifdef __cplusplus
#define SETUP_PROJECT_DEF extern "C"{	\
EAGLE_EXPORTS void SetupIncludes();		\
EAGLE_EXPORTS int GetBuildType();		\
}
#else
#define SETUP_PROJECT_DEF EAGLE_EXPORTS void SetupIncludes();
#endif


// *************** SETUP_PROJECT_IMPL ********************
#ifndef PROJECT_ID
#define PROJECT_ID 0
#endif
#ifndef PROJECT_INCLUDES
#define PROJECT_INCLUDES ""
#endif
#ifndef PROJECT_LIB_DIRS
#define PROJECT_LIB_DIRS ""
#endif
#ifndef PROJECT_DEFINITIONS
#define PROJECT_DEFINITIONS ""
#endif
#ifndef PROJECT_CONFIG_FILE
#define PROJECT_CONFIG_FILE ""
#endif
#ifndef BUILD_TYPE
	#ifdef _DEBUG
		#define BUILD_TYPE 0
	#else
		#define BUILD_TYPE 1
	#endif
#endif

#define SETUP_PROJECT_IMPL	int GetBuildType() {return BUILD_TYPE; }									\
void SetupIncludes(){																					\
		int id = EagleLib::NodeManager::getInstance().parseProjectConfig(PROJECT_CONFIG_FILE);			\
		PerModuleInterface::GetInstance()->SetProjectIdForAllConstructors(id);							\
		EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES, id);						\
		EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS, id);							\
		EagleLib::NodeManager::getInstance().addDefinitions(PROJECT_DEFINITIONS, id);					\
}																										

namespace EagleLib
{
	enum PlaybackState
	{
		PLAYING,
		PAUSED,
		FAST_FORWARD,
		FAST_BACKWARD,
		STOP
	};
}