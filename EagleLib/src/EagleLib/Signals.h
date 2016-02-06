#pragma once
#include <boost/preprocessor.hpp>
#include "EagleLib/Defs.hpp"
#include "signals/signal_manager.h"
#include <EagleLib/rcc/SystemTable.hpp>
#include <ObjectInterfacePerModule.h>

#define SIG_DEF_1(name) \
inline void sig_##name() \
{ \
    static auto registerer = EagleLib::register_sender<void(void), -1>(this, #name); \
    registerer(); \
}

#define SIG_DEF_2(name, ARG1) \
inline void sig_##name(ARG1 arg1) \
{ \
    static auto registerer = register_sender<void(ARG1), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1)))); \
    registerer(arg1); \
}

#define SIG_DEF_3(name, ARG1, ARG2) \
inline void sig_##name(ARG1 arg1, ARG2 arg2) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2)))); \
    registerer(arg1, arg2); \
}

#define SIG_DEF_4(name, ARG1, ARG2, ARG3) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3)))); \
    registerer( arg1, arg2, arg3); \
}

#define SIG_DEF_5(name, ARG1, ARG2, ARG3, ARG4) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4)))); \
    registerer(arg1, arg2, arg3, arg4); \
}

#define SIG_DEF_6(name, ARG1, ARG2, ARG3, ARG4, ARG5) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4, ARG5), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5)))); \
    registerer( arg1, arg2, arg3, arg4, arg5); \
}

#define SIG_DEF_7(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6)))); \
    registerer( arg1, arg2, arg3, arg4, arg5, arg6); \
}

#define SIG_DEF_8(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7)))); \
    registerer( arg1, arg2, arg3, arg4, arg5, arg6, arg7); \
}

#define SIG_DEF_9(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8))));\
    registerer( arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); \
}

#define SIG_DEF_10(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8, ARG9 arg9) \
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8,ARG9)))); \
    registerer( arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); \
}

#define SIG_DEF_11(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10) \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8, ARG9 arg9, ARG10 arg10)\
{ \
    static auto registerer = register_sender<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10), -1>(this, #name, Loki::TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8,ARG9,ARG10)))); \
    registerer( arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10); \
}

// name / signature
#ifdef _MSC_VER
#define SIG_DEF(...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(SIG_DEF_, __VA_ARGS__ )(__VA_ARGS__), BOOST_PP_EMPTY() )
#else
#define SIG_DEF(...) BOOST_PP_OVERLOAD(SIG_DEF_, __VA_ARGS__ )(__VA_ARGS__)
#endif


namespace EagleLib
{

    

    // Basically reimplements some of the stuff from Signals::signal_manager, but with
    // access restrictions and stream segregation
    class EAGLE_EXPORTS SignalManager: public Signals::signal_manager
    {
    public:
        template<typename T, typename C> std::shared_ptr<Signals::connection> Connect(const std::string& name, const std::function<T>& f, C* receiver, int stream_index = -1, boost::thread::id dest_thread = boost::this_thread::get_id())
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto&sig = GetSignal(name, Loki::TypeInfo(typeid(T)), stream_index);
            Signals::signal<T>* typedSignal = nullptr;
            if (!sig)
            {
                typedSignal = new Signals::signal<T>();
                sig.reset(typedSignal);
            }else
            {
                typedSignal = std::dynamic_pointer_cast<Signals::signal<T>>(sig).get();
            }
            ConnectionInfo cinfo;
            cinfo.ptr = (void*)receiver;
            cinfo.sender = false;
            cinfo.signal_name = name;
            cinfo.object_type = Loki::TypeInfo(typeid(C));
            cinfo.signal_signature = Loki::TypeInfo(typeid(T));
            connections.push_back(cinfo);
            return typedSignal->connect(f, dest_thread);
        }


        template<typename T, typename C> Signals::signal<T>* GetSignal(const std::string& name, C* sender, int stream_index = -1)
        {
            std::lock_guard<std::mutex> lock(mtx);

            auto&sig = GetSignal(name, Loki::TypeInfo(typeid(T)), stream_index);
            if (!sig)
                sig.reset(new Signals::signal<T>());

            ConnectionInfo cinfo;
            cinfo.sender = true;
            cinfo.ptr = (void*)sender;
            cinfo.signal_name = name;
            cinfo.object_type = Loki::TypeInfo(typeid(C));
            cinfo.signal_signature = Loki::TypeInfo(typeid(T));
            connections.push_back(cinfo);
            return std::dynamic_pointer_cast<Signals::signal<T>>(sig).get();
        }

    private:
        struct ConnectionInfo
        {
            bool sender; // True if sender, false if receiver
            Loki::TypeInfo object_type;
            Loki::TypeInfo signal_signature;
            void* ptr;
            std::string signal_name;
        };
        std::shared_ptr<Signals::signal_base>& GetSignal(const std::string& name, Loki::TypeInfo type, int stream_index);
        // The signal manager object stores signals by the stream index, signature, and name
        std::map<int,  // Stream index
        std::map<Loki::TypeInfo, // Signal signature
        std::map<std::string, // Signal name
            std::shared_ptr<Signals::signal_base> >>> stream_specific_signals;
        
        /*std::map<Loki::TypeInfo, // Signal signature
        std::map<std::string, // Signal name
            std::shared_ptr<Signals::signal_base> >> global_signals;*/

        std::list<ConnectionInfo> connections;
        std::mutex mtx;
    
    };

    // Class that can send and receive signals

    template<typename T, int S_ID> class register_sender
    {
    };
    template<typename R, typename... T, int S_ID> struct register_sender<R(T...), S_ID>
    {
        template<typename C> register_sender(C* sender, const std::string& signal_name)
        {
            auto table = PerModuleInterface::GetInstance()->GetSystemTable();
            auto manager = table->GetSingleton<EagleLib::SignalManager>();
            signal = manager->GetSignal<R(T...)>(signal_name, sender, S_ID);
        }
        void operator()(T... args)
        {
            (*signal)(args...);
        }
        ~register_sender()
        {
            // todo remove signal from signal manager
        }
        Signals::signal<R(T...)>* signal;
    };
}
