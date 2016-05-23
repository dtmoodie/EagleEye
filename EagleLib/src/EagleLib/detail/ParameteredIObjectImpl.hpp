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


#define BEGIN_PARAM_1(DERIVED, N_) \
protected: \
	typedef DERIVED PARAM_THIS_CLASS; \
	typedef ParameteredIObject PARAM_PARENT; \
	void call_parent(ISimpleSerializer* pSerializer){ ParameteredIObject::SerializeAllParams(pSerializer); } \
	void init_parent() {ParameteredIObject::RegisterAllParams(); } \
	template<int N> void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
	{ \
		SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
	} \
	template<int N> void InitializeParams(Signals::_counter_<N> dummy) \
	{ \
		InitializeParams(Signals::_counter_<N-1>()); \
	} \
	void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){} \
	void InitializeParams(Signals::_counter_<N_> dummy){}

#define BEGIN_PARAM_2(DERIVED, BASE, N) \
	typedef DERIVED PARAM_THIS_CLASS; \
	typedef BASE PARAM_PARENT; \
	void call_parent(ISimpleSerializer* pSerializer){ BASE::SerializeParams(pSerializer); } \
	void init_parent() {BASE::RegisterAllParams(); } \
	template<int N> void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
	{ \
		SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
	} \
	template<int N> void InitializeParams(Signals::_counter_<N> dummy) \
	{ \
		InitializeParams(Signals::_counter_<N-1>()); \
	} \
	void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){} \
	void InitializeParams(Signals::_counter_<N_> dummy){}



#define END_PARAMS_(N) \
public: \
	void SerializeAllParams(ISimpleSerializer* pSerializer) \
	{ \
		call_parent(pSerializer); \
		SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
	} \
	void RegisterAllParams() \
	{ \
		init_parent(); \
		InitializeParams(Signals::_counter_<N-1>()); \
	}

#define PARAM_(type, name, init, N) \
	type name = init; \
	Parameters::TypedParameterPtr<type> name##_param; \
	void InitializeParams(Signals::_counter_<N> dummy) \
	{ \
		name##_param.SetName(#name); \
		name##_param.UpdateData(&name); \
		ParameteredObject::addParameter(&name##_param); \
		InitializeParams(Signals::_counter_<N-1>()); \
	} \
	void SerializeParams(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
	{ \
		SERIALIZE(name); \
		SerializeParams(pSerializer, Signals::_counter_<N-1>()); \
	}

#define RANGED_PARAM_(type, name, init, min, max, N) \
	type name = init; \
	Parameters::RangedParameterPtr<type> name##_param; \
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