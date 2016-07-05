#pragma once
#include "parameters/ParameteredObjectImpl.hpp"
#undef BEGIN_PARAMS
#undef DEFINE_PARAM_3
#undef DEFINE_PARAM_4
#undef DEFINE_PARAM_5
#undef DEFINE_PARAM_6
#undef PARAM_2
#undef PARAM_3
#undef PARAM_4
#undef PARAM_5
#undef PARAM
#undef END_PARAMS_
#undef END_PARAMS


#define BEGIN_PARAMS__1(DERIVED, N_) \
    BEGIN_PARAMS_1(DERIVED, N_); \
    template<int N> void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    } \
    void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){}


#define BEGIN_PARAMS__2(DERIVED, BASE, N) \
    BEGIN_PARAMS_2(DERIVED, BASE, N) \
    template<int N> void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    } \
    void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){} \
    



#define PARAM_(type, name, init, N) \
    DEFINE_PARAM_4(type, name, initial_value, N); \
    type name; \
    Parameters::TypedParameterPtr<type> name##_param; \
    void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SERIALIZE(name); \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    }

#define RANGED_PARAM_(type, name, init, min, max, N) \
    DEFINE_PARAM_5(type, name, min, max, N); \
    type name; Parameters::RangedParameterPtr<type> name##_param; \
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
        call_parent(pSerializer); \
        SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
    }