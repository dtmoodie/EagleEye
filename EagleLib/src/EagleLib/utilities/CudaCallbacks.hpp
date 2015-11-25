#pragma once
#include "EagleLib/Defs.hpp"
#include <opencv2/core/cuda.hpp>
#include <functional>
#include <future>

#include <pplx/pplxtasks.h>

namespace EagleLib
{
	namespace cuda
	{

		struct EAGLE_EXPORTS scoped_event_timer
		{
			std::string _scope_name;
			cv::cuda::Stream _stream;
			cv::cuda::Event _start;
			cv::cuda::Event _end;
			scoped_event_timer(cv::cuda::Stream& stream, const std::string& scope_name = "");
			~scoped_event_timer();
		};
		struct EAGLE_EXPORTS ICallback
		{
			static void cb_func(int status, void* user_data);
			virtual void run() = 0;
		};
		
		template<typename T, typename C> 
		auto enqueue_callback(
			const T& user_data, 
			cv::cuda::Stream& stream) -> void
		{
			static_assert(std::is_base_of<ICallback, C>::value, "Template class argument must inherit from ICallback");
			stream.enqueueHostCallback(ICallback::cb_func, new C(user_data));
		}

		
		
		template<typename _return_type> 
		struct FunctionCallback: public ICallback
		{
			std::function<_return_type()> func;
			std::promise<_return_type> promise;
			FunctionCallback(const std::function<_return_type()>& f) : func(f) {}
			virtual void run();
		};


		template<typename _Ty> auto
			enqueue_callback(_Ty function,cv::cuda::Stream& stream)->std::future<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>
		{
			auto fc = new FunctionCallback<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>(function);
			stream.enqueueHostCallback(&ICallback::cb_func, fc);
			return fc->promise.get_future();
		}
		


		template<typename T> class Callback: public ICallback
		{
			Callback(const T& data);
			virtual void run();
		};


		// Implementations
		template<typename T> void FunctionCallback<T>::run()
		{
			promise.set_value(func());
		}

		template<typename T> void Callback<T>::run()
		{

		}
	}
}



