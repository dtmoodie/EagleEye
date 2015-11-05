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
#define SETUP_PROJECT_DEF extern "C"{ EAGLE_EXPORTS void SetupIncludes(); }
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
/*#define SETUP_PROJECT_IMPL void SetupIncludes(){ 														\
		EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES, PROJECT_ID);				\
		EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS, PROJECT_ID);					\
		EagleLib::NodeManager::getInstance().addDefinitions(PROJECT_DEFINITIONS, PROJECT_ID);			\
}*/
#define SETUP_PROJECT_IMPL void SetupIncludes(){														\
		int id = EagleLib::NodeManager::getInstance().parseProjectConfig(PROJECT_CONFIG_FILE);			\
		PerModuleInterface::GetInstance()->SetProjectIdForAllConstructors(id);							\
		EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES, id);						\
		EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS, id);							\
		EagleLib::NodeManager::getInstance().addDefinitions(PROJECT_DEFINITIONS, id);					\
}																										



