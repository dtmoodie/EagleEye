#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define CU_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CU_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define CU_EXPORTS
#endif

#ifdef _MSC_VER
#ifndef cuda_utils_EXPORTS
#ifdef _DEBUG
#pragma comment(lib, "cuda_utilsd.lib")
#else
#pragma comment(lib, "cuda_utils.lib")
#endif
#endif
#endif