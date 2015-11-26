#pragma once
#include "EagleLib/Defs.hpp"
#include <opencv2/core/cuda.hpp>
#include <functional>
#include <future>

#include <pplx/pplxtasks.h>
#include <boost/log/trivial.hpp>
#include <iostream>
namespace EagleLib
{
	namespace cuda
	{

		struct EAGLE_EXPORTS scoped_event_timer
		{
			std::string _scope_name;
			cv::cuda::Stream _stream;
			clock_t start;
			scoped_event_timer(cv::cuda::Stream& stream, const std::string& scope_name = "");
			~scoped_event_timer();
		};
		struct EAGLE_EXPORTS ICallback
		{
			static void cb_func_async(int status, void* user_data);
			static void cb_func(int status, void* user_data);
			virtual ~ICallback();
			virtual void run() = 0;
		};
		
		template<typename T, typename C> 
		auto enqueue_callback_async(
			const T& user_data, 
			cv::cuda::Stream& stream) -> void
		{
			static_assert(std::is_base_of<ICallback, C>::value, "Template class argument must inherit from ICallback");
			stream.enqueueHostCallback(ICallback::cb_func_async, new C(user_data));
		}
		template<typename T, typename C>
		auto enqueue_callback(
			const T& user_data,
			cv::cuda::Stream& stream) -> void
		{
			static_assert(std::is_base_of<ICallback, C>::value, "Template class argument must inherit from ICallback");
			stream.enqueueHostCallback(ICallback::cb_func, new C(user_data));
		}


		template<typename T, typename R>
		struct FunctionCallback : public ICallback
		{
			std::function<R(T)> func;
			T data;
			std::promise<R> promise;
			
			FunctionCallback(const T& d, const std::function<R(T)> f) : func(f), data(d) {}
			virtual ~FunctionCallback() {}
			virtual void run();
		};
		
		template<typename _return_type> 
		struct LambdaCallback: public ICallback
		{
			std::function<_return_type()> func;
			std::promise<_return_type> promise;
			
			LambdaCallback(const std::function<_return_type()>& f) : func(f) {}
			~LambdaCallback() {}
			virtual void run();
		};

		// While this does technically work, for some reason it takes significantly more
		// Gpu time to execute given the same callback.
		template<typename _Ty> auto
			enqueue_callback_async(_Ty function,cv::cuda::Stream& stream)->std::future<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>
		{
			auto fc = new LambdaCallback<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>(function);
			stream.enqueueHostCallback(&ICallback::cb_func_async, fc);
			return fc->promise.get_future();
		}
		template<typename _Ty> auto
			enqueue_callback(_Ty function, cv::cuda::Stream& stream)->std::future<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>
		{
			auto fc = new LambdaCallback<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>(function);
			stream.enqueueHostCallback(&ICallback::cb_func, fc);
			return fc->promise.get_future();
		}


		template<typename T, typename R> std::future<R>
			enqueue_callback_async(const T& data, const std::function<R(T)>& function, cv::cuda::Stream& stream)
		{
			auto fc = new FunctionCallback<T,R>(data, function);
			stream.enqueueHostCallback(&ICallback::cb_func_async, fc);
			return fc->promise.get_future();
		}

		template<typename T, typename R> std::future<R>
			enqueue_callback(const T& data, const std::function<R(T)>& function, cv::cuda::Stream& stream)
		{
			auto fc = new FunctionCallback<T, R>(data, function);
			stream.enqueueHostCallback(&ICallback::cb_func, fc);
			return fc->promise.get_future();
		}

		template<typename T> class Callback: public ICallback
		{
			Callback(const T& data);
			virtual void run();
		};


		// Implementations
		template<typename T> void LambdaCallback<T>::run()
		{
			promise.set_value(func());
		}
		template<typename T, typename R> void FunctionCallback<T, R>::run()
		{
			promise.set_value(func(data));
		}

		template<typename T> void Callback<T>::run()
		{

		}
	}
}



