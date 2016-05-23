#pragma once
#include "EagleLib/detail/ParameteredIObjectImpl.hpp"

#ifdef _MSC_VER
#define BEGIN_PARAMS(...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(BEGIN_PARAM_, __VA_ARGS__ )(__VA_ARGS__, __COUNTER__), BOOST_PP_EMPTY() )
#else
#define BEGIN_PARAMS(...) BOOST_PP_OVERLOAD(BEGIN_PARAM_, __VA_ARGS__ )(__VA_ARGS__, __COUNTER__)
#endif

#define PARAM(type, name, init) PARAM_(type, name, init, __COUNTER__);
#define RANGED_PARAM(type, name, init, min, max) RANGED_PARAM_(type, name, init, min, max, __COUNTER__);

#define END_PARAMS END_PARAMS_(__COUNTER__);