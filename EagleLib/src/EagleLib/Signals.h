#pragma once
#include <boost/preprocessor.hpp>
#include "EagleLib/Defs.hpp"
#include "signals/signal_manager.h"
#include "signals/signal.h"
#include <EagleLib/rcc/SystemTable.hpp>
#include <ObjectInterfacePerModule.h>
/*
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

*/
namespace EagleLib
{

    

    // Basically reimplements some of the stuff from Signals::signal_manager, but with
    // access restrictions and stream segregation
    class EAGLE_EXPORTS SignalManager: public Signals::signal_manager
    {
    public:
		/*template<typename T, 
			template<class> class combiner = Signals::default_combiner, 
			template<class...> class Sink = Signals::signal_sink> 
		Signals::typed_signal_base<T, combiner>* get_signal(const std::string& name, const std::string& description = "", std::string file_name = "", int line_number = -1)
		{
			std::lock_guard<std::mutex> lock(mtx);
			if (line_number != -1 && file_name.size())
				register_sender(Loki::TypeInfo(typeid(T)), name, description, file_name, line_number);

			auto&sig = get_signal(name, Loki::TypeInfo(typeid(Signals::typed_signal_base<T, combiner>)));
			if (!sig)
				sig.reset(new Signals::typed_signal<T, combiner, Sink>(description));

			return std::dynamic_pointer_cast<typed_signal_base<T, combiner>>(sig).get();
		}

		template<typename T, template<class> class combiner = Signals::default_combiner> std::shared_ptr<Signals::connection> connect(const std::string& name, std::function<T> f, boost::thread::id destination_thread = boost::this_thread::get_id(), const std::string& receiver_description = "", int line_number = -1, const std::string& filename = "")
		{
			auto sig = get_signal<T, combiner>(name);
			if (filename.size() && line_number != -1)
				register_receiver(Loki::TypeInfo(typeid(T)), name, line_number, filename, receiver_description);
			return sig->connect(f, destination_thread);
		}
		template<typename T, typename C> std::shared_ptr<Signals::connection> connect(const std::string& name, std::function<T> f, C* receiver, boost::thread::id destination_thread = boost::this_thread::get_id(), const std::string& receiver_description = "")
		{
			auto sig = get_signal<T>(name);
			register_receiver(Loki::TypeInfo(typeid(T)), name, Loki::TypeInfo(typeid(C)), receiver);
			return sig->connect(f, destination_thread);
		}*/
		static	SignalManager* get_instance();
    };


}
