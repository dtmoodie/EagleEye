#pragma once

// Guard this file from being compiled by nvcc
#ifndef __CUDACC__
#include "Export.hpp"
#include "IRuntimeObjectSystem.h"
#endif

#define TOKEN_TO_STRING_(token) #token
#define TOKEN_TO_STRING(token) TOKEN_TO_STRING_(token)


#ifdef PLUGIN_NAME
  #include "RuntimeLinkLibrary.h"
  #if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
    #define PLUGIN_EXPORTS __declspec(dllexport)
    #ifdef _DEBUG
      RUNTIME_COMPILER_LINKLIBRARY(TOKEN_TO_STRING(PLUGIN_NAME) "d.lib")
    #else
      RUNTIME_COMPILER_LINKLIBRARY(TOKEN_TO_STRING(PLUGIN_NAME) ".lib")
    #endif
  #else // Linux
    #define PLUGIN_EXPORTS
    #ifdef _DEBUG
      //RUNTIME_COMPILER_LINKLIBRARY("-l" TOKEN_TO_STRING(PLUGIN_NAME) )
    #else
      //RUNTIME_COMPILER_LINKLIBRARY("-l" TOKEN_TO_STRING(PLUGIN_NAME) "d")
    #endif
  #endif
#else // PLUGIN_NAME
  #define PLUGIN_EXPORTS
#endif

#ifndef PLUGIN_EXPORTS
  #define PLUGIN_EXPORTS
#endif


// *************** SETUP_PROJECT_DEF ********************
/*#ifdef __cplusplus
  #define SETUP_PROJECT_DEF extern "C"{    \
    PLUGIN_EXPORTS void SetupIncludes();   \
    PLUGIN_EXPORTS int GetBuildType();     \
  }
#else
  #define SETUP_PROJECT_DEF PLUGIN_EXPORTS void SetupIncludes(); PLUGIN_EXPORTS int GetBuildType();
#endif*/
#define SETUP_PROJECT_DEF


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

#define SETUP_PROJECT_IMPL
/*#define SETUP_PROJECT_IMPL    int GetBuildType() {return BUILD_TYPE; }                                  \
void SetupIncludes(){                                                                                   \
        auto id = mo::MetaObjectFactory::Instance()->GetObjectSystem()->ParseConfigFile(PROJECT_CONFIG_FILE);     \
        PerModuleInterface::GetInstance()->SetProjectIdForAllConstructors(id);                          \
}*/

