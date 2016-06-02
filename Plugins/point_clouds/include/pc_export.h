#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined point_clouds_EXPORTS
#define PC_EXPORTS __declspec(dllexport)
#else // linux
#define PC_EXPORTS 

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("-lpoint_clouds" )
#else
RUNTIME_COMPILER_LINKLIBRARY("-lpoint_cloudsd")
#endif

#endif

#ifndef PC_EXPORTS
#define PC_EXPORTS 
#endif