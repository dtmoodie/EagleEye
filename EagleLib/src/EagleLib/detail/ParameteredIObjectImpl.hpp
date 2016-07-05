#pragma once
#include "parameters/ParameteredObjectImpl.hpp"
#undef PARAM
#undef END_PARAMS
#undef BEGIN_PARAMS

#define VA_ARGS(...) , ##__VA_ARGS__




#define PARAM__2(N, type, name) \
Parameters::TypedParameterPtr<type> name##_param; \
void WrapParams_(Signals::_counter_<N> dummy) \
{ \
    name##_param.SetName(#name); \
    name##_param.UpdateData(&name); \
    ParameteredObject::addParameter(&name##_param); \
    WrapParams_(--dummy); \
} \
static void getParameterInfo_(std::vector<Parameters::ParameterInfo*>& info, Signals::_counter_<N> dummy) \
{ \
    static Parameters::ParameterInfo s_info{Loki::TypeInfo(typeid(type)), #name}; \
    info.push_back(&s_info); \
    getParameterInfo_(info, --dummy); \
} \
void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
{ \
    SERIALIZE(name); \
    SerializeParams(pSerializer, --dummy); \
}

#define PARAM__3(N, type, name, initial_value) \
Parameters::TypedParameterPtr<type> name##_param; \
void InitParams_(Signals::_counter_<N> dummy) \
{ \
    name = type(initial_value); \
    InitParams_(--dummy); \
} \
void WrapParams_(Signals::_counter_<N> dummy) \
{ \
    name##_param.SetName(#name); \
    name##_param.UpdateData(&name); \
    ParameteredObject::addParameter(&name##_param); \
    WrapParams_(--dummy); \
} \
static void getParameterInfo_(std::vector<Parameters::ParameterInfo*>& info, Signals::_counter_<N> dummy) \
{ \
    static Parameters::ParameterInfo s_info{Loki::TypeInfo(typeid(type)), #name}; \
    info.push_back(&s_info); \
    getParameterInfo_(info, --dummy); \
} \
void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
{ \
    SERIALIZE(name); \
    SerializeParams(pSerializer, --dummy); \
}

#define PARAM__4(N, type, name, min, max) \
Parameters::RangedParameterPtr<type> name##_param; \
void WrapParams_(Signals::_counter_<N> dummy) \
{ \
    name##_param.SetName(#name); \
    name##_param.UpdateData(&name); \
    name##_param.SetRange(min, max); \
    ParameteredObject::addParameter(&name##_param); \
    WrapParams_(--dummy); \
} \
static void getParameterInfo_(std::vector<Parameters::ParameterInfo*>& info, Signals::_counter_<N> dummy) \
{ \
    static Parameters::ParameterInfo s_info{Loki::TypeInfo(typeid(type)), #name}; \
    info.push_back(&s_info); \
    getParameterInfo_(info, --dummy); \
} \
void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
{ \
    SERIALIZE(name); \
    SerializeParams(pSerializer, --dummy); \
}

#define PARAM__5(N, type, name, min, max, initial_value) \
Parameters::RangedParameterPtr<type> name##_param; \
void InitParams_(Signals::_counter_<N> dummy) \
{ \
    name = type(initial_value); \
    InitParams_(Signals::_counter_<N-1>()); \
} \
void WrapParams_(Signals::_counter_<N> dummy) \
{ \
    name##_param.SetName(#name); \
    name##_param.UpdateData(&name); \
    name##_param.SetRange(min, max); \
    ParameteredObject::addParameter(&name##_param); \
    WrapParams_(--dummy); \
} \
static void getParameterInfo_(std::vector<Parameters::ParameterInfo*>& info, Signals::_counter_<N> dummy) \
{ \
    static Parameters::ParameterInfo s_info{Loki::TypeInfo(typeid(type)), #name}; \
    info.push_back(&s_info); \
    getParameterInfo_(info, --dummy); \
} \
void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
{ \
    SERIALIZE(name); \
    SerializeParams(pSerializer, --dummy); \
}

#ifdef _MSC_VER
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


#define BEGIN_PARAMS__1(DERIVED, N_) \
    BEGIN_PARAMS_1(DERIVED, N_); \
    void SerializeParentParams(ISimpleSerializer* pSerializer) {} \
    template<int N> void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    } \
    void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){}


#define BEGIN_PARAMS__2(DERIVED, BASE, N) \
    BEGIN_PARAMS_2(DERIVED, BASE, N) \
    void SerializeParentParams(ISimpleSerializer* pSerializer) { BASE::SerializeParams(pSerializer); } \
    template<int N> void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    } \
    void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){} \
    

#define RANGED_PARAM_(type, name, init, min, max, N) \
    type name; Parameters::RangedParameterPtr<type> name##_param; \
    DEFINE_PARAM_5(type, name, min, max, N); \
    void InitializeParams(Signals::_counter_<N> dummy) \
    { \
        name##_param.SetName(#name); \
        name##_param.UpdateData(&name); \
        name##_param.SetRange(min, max); \
        ParameteredObject::addParameter(&name##_param); \
        InitializeParams(Signals::_counter_<N-1>()); \
    } \
    void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SERIALIZE(name); \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    }

#define END_PARAMS__(N) \
    END_PARAMS_(N); \
public: \
    void SerializeAllParams(ISimpleSerializer* pSerializer) \
    { \
        SerializeParentParams(pSerializer); \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    }