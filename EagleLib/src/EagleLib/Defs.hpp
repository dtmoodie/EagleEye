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
#ifdef PROJECT_INCLUDES
#ifdef PROJECT_LIB_DIRS
#define SETUP_PROJECT_IMPL void SetupIncludes(){ 																	\
				EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES);										\
				EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS);}		
#else
#define SETUP_PROJECT_IMPL void SetupIncludes(){EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES);}		
#endif
#else
#ifdef PROJECT_LIB_DIRS
#define SETUP_PROJECT_IMPL void SetupIncludes()	{EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS);}
#else
#ifndef EagleLIB_EXPORTS
//#pragma message( "Neither PROJECT_LIB_DIRS nor PROJECT_INCLUDES defined" )
#endif
#define SETUP_PROJECT_IMPL void SetupIncludes() {}
#endif
#endif