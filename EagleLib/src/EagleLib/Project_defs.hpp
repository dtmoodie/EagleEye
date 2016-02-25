#pragma once
#include "EagleLib/rcc/ObjectManager.h"
#include "Defs.hpp"

#ifdef PLUGIN_NAME
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows
#define PLUGIN_EXPORTS __declspec(dllexport) 
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY(STRINGIFY(PLUGIN_NAME) "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY(STRINGIFY(PLUGIN_NAME) ".lib")
#endif

#else // Linux
#define PLUGIN_EXPORTS
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("-l" STRINGIFY(PLUGIN_NAME) )
#else
RUNTIME_COMPILER_LINKLIBRARY("-l" STRINGIFY(PLUGIN_NAME) "d")
#endif
#endif
#else
#define PLUGIN_EXPORTS
#endif

#ifndef PLUGIN_EXPORTS
#define PLUGIN_EXPORTS
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
		int id = EagleLib::ObjectManager::Instance().parseProjectConfig(PROJECT_CONFIG_FILE);			\
		PerModuleInterface::GetInstance()->SetProjectIdForAllConstructors(id);							\
		EagleLib::ObjectManager::Instance().addIncludeDirs(PROJECT_INCLUDES, id);						\
		EagleLib::ObjectManager::Instance().addLinkDirs(PROJECT_LIB_DIRS, id);							\
		EagleLib::ObjectManager::Instance().addDefinitions(PROJECT_DEFINITIONS, id);					\
}
