#pragma once
#include "EagleLib/Defs.hpp"
#include "IObject.h"
#include <parameters/ParameteredObject.h>

#undef SIG_SEND
#undef SIGNALS_BEGIN
#undef SIGNALS_END

#ifdef _MSC_VER
// MSVC overloaded signals
#define SERIALIZE_SIGNAL(N, name, ...) \
    void SerializeSignals(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SERIALIZE(COMBINE(_sig_##name##_, N)); \
        SerializeSignals(pSerializer, Signals::_counter_<N-1>()); \
    }

#define SIG_SEND__(N, ...) \
    BOOST_PP_CAT( BOOST_PP_OVERLOAD(SIGNAL_, __VA_ARGS__ )(__VA_ARGS__, N), BOOST_PP_EMPTY() ) \
    //SERIALIZE_SIGNAL(N, __VA_ARGS__)
    

#define SIGNALS_BEGIN__(N_, ...)    \
    BOOST_PP_CAT(BOOST_PP_OVERLOAD(SIGNALS_BEGIN_, __VA_ARGS__)(__VA_ARGS__, N_), BOOST_PP_EMPTY()) \
    template<int N> void SerializeSignals(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SerializeSignals(pSerializer, Signals::_counter_<N-1>()); \
    } \
    void SerializeSignals(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){}

#else

#define SERIALIZE_SIGNAL(N, name, ...) \
    void SerializeSignals(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SERIALIZE(COMBINE(_sig_##name##_, N)); \
        SerializeSignals(pSerializer, Signals::_counter_<N-1>()); \
    }

#define SIG_SEND__(N, ...) \
    BOOST_PP_OVERLOAD(SIGNAL_, __VA_ARGS__ )(__VA_ARGS__, N) \
    SERIALIZE_SIGNAL(N, __VA_ARGS__)

#define SIGNALS_BEGIN__(N_, ...)    \
    BOOST_PP_OVERLOAD(SIGNALS_BEGIN_, __VA_ARGS__)(__VA_ARGS__, N_) \
    template<int N> void SerializeSignals(ISimpleSerializer* pSerializer, Signals::_counter_<N> dummy) \
    { \
        SerializeSignals(pSerializer, Signals::_counter_<N-1>()); \
    } \
    void SerializeSignals(ISimpleSerializer* pSerializer, Signals::_counter_<N_> dummy){}
#endif


#define SIGNALS_END__(N) \
    SIGNALS_END_(N) \
    virtual void SerializeAllSignals(ISimpleSerializer* pSerializer) \
    { \
        EagleLib::_serialize_parent_signals<THIS_CLASS>(pSerializer, this, 0); \
        SerializeSignals(pSerializer, Signals::_counter_<N-1>()); \
    }
    


// main entry point macros
#define SIG_SEND(...) SIG_SEND__(__COUNTER__, __VA_ARGS__)
#define SIGNALS_BEGIN(...) SIGNALS_BEGIN__(__COUNTER__, __VA_ARGS__)
#define SIGNALS_END SIGNALS_END__(__COUNTER__)





namespace EagleLib
{
    // This is chosen in the case that the class correctly defines
    template<class T> auto _serialize_parent_signals_helper(ISimpleSerializer* pSerializer, T* obj, int) ->decltype(obj->SerializeAllSignals(pSerializer), void())
    {
        obj->SerializeAllSignals(pSerializer);
    }
    // This is chosen if a parent class exists but does not have a serialize signals function
    template<class T> void _serialize_parent_signals_helper(ISimpleSerializer* pSerializer, T* obj, long)
    {
    }
    template<class T> void _serialize_parent_signals(ISimpleSerializer* pSerializer, T* This, typename std::enable_if<Signals::has_parent<T>::value, int>::type)
    {
        _serialize_parent_signals_helper<typename T::PARENT_CLASS>(pSerializer, This, 0);
    }
    // Class doesn't even define it's parent class, thus no reason to check if parent has a serialize signals function
    template<class T> void _serialize_parent_signals(ISimpleSerializer* pSerializer, T* This, long) 
    {
    }

    class EAGLE_EXPORTS ParameteredIObject : public IObject, public Parameters::ParameteredObject
    {
    public:
        ParameteredIObject();
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(const cv::FileNode& configNode);
        virtual void Init(bool firstInit);
        virtual void SerializeAllParams(ISimpleSerializer* pSerializer);

        SIGNALS_BEGIN(ParameteredIObject, ParameteredObject)
            SIG_SEND(object_recompiled, ParameteredIObject*)
        SIGNALS_END
    protected:
    };
}
