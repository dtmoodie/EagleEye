#pragma once
#include <boost/preprocessor.hpp>
#include "RuntimeLinkLibrary.h"
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
#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("EagleLibd.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("EagleLib.lib");
#endif
#else

#endif

																									

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