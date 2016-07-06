#pragma once
#include "EagleLib/detail/ParameteredIObjectImpl.hpp"

#ifdef _MSC_VER
#define BEGIN_PARAMS(...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(BEGIN_PARAMS__, __VA_ARGS__ )(__VA_ARGS__, __COUNTER__), BOOST_PP_EMPTY() )
#else
#define BEGIN_PARAMS(...) BOOST_PP_OVERLOAD(BEGIN_PARAMS__, __VA_ARGS__ )(__VA_ARGS__, __COUNTER__)
#endif

#ifdef _MSC_VER
// While this does work, it breaks intellisense which is really annoying
//#define PARAM(TYPE, NAME, ...)  \
//    TYPE NAME; \
//    BOOST_PP_CAT( BOOST_PP_OVERLOAD(PARAM__, TYPE, NAME VA_ARGS(__VA_ARGS__) )(__COUNTER__, TYPE, NAME, ##__VA_ARGS__), BOOST_PP_EMPTY() )

#define PARAM(TYPE, NAME, INIT) \
    TYPE NAME = TYPE(INIT); \
    PARAM__3(__COUNTER__, TYPE, NAME, INIT)
                
#else
#define PARAM(...) BOOST_PP_OVERLOAD(PARAM_, __VA_ARGS__ )(__VA_ARGS__) \
                   BOOST_PP_OVERLOAD(SERIALIZE_PARAM_, __VA_ARGS__ )(__VA_ARGS__)
#endif

#define RANGED_PARAM(type, name, init, min, max) RANGED_PARAM_(type, name, init, min, max, __COUNTER__);

#define END_PARAMS END_PARAMS__(__COUNTER__);