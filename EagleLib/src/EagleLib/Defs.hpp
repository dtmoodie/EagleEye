#pragma once

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
	enum NodeType
	{
		Source			= 1 << 1,
		Sink			= 1 << 2,
		Processing		= 1 << 3,
		Extractor		= 1 << 4,


		// Datatypes that nodes operate on
		Image			= 1 << 16,
		PtCloud         = 1 << 17,
		Tensor          = 1 << 18
	};
}

