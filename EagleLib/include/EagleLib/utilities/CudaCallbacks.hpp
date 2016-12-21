#pragma once
#include "EagleLib/Detail/Export.hpp"
#include "ObjectPool.hpp"
#include <opencv2/core/cuda.hpp>
#include <pplx/pplxtasks.h>
#include <boost/log/trivial.hpp>
#include <iostream>
#include <functional>
#include <future>
namespace boost
{
    namespace posix_time
    {
        class ptime;
    }
}
namespace EagleLib
{
    namespace cuda
    {

        struct EAGLE_EXPORTS scoped_stream_timer
        {
            std::string _scope_name;
            cv::cuda::Stream _stream;
            boost::posix_time::ptime* start_time;
            scoped_stream_timer(cv::cuda::Stream& stream, const std::string& scope_name = "");
            ~scoped_stream_timer();
        };
        struct EAGLE_EXPORTS scoped_event_stream_timer
        {
            static EagleLib::pool::ObjectPool<cv::cuda::Event> eventPool;
            EagleLib::pool::Ptr<cv::cuda::Event> startEvent;
            EagleLib::pool::Ptr<cv::cuda::Event> endEvent;
            std::string _scope_name;
            cv::cuda::Stream _stream;
            scoped_event_stream_timer(cv::cuda::Stream& stream, const std::string& scope_name = "");
            ~scoped_event_stream_timer();
        };

        struct EAGLE_EXPORTS ICallback
        {
            static void cb_func_async_event_loop(int status, void* user_data);
            static void cb_func_async(int status, void* user_data);
            static void cb_func(int status, void* user_data);
            
            virtual ~ICallback();
            virtual void run() = 0;
        };
        struct EAGLE_EXPORTS ICallbackEventLoop: virtual public ICallback
        {
            size_t event_loop_thread_id;
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
        template<typename T> struct FunctionCallback<T, void>: public ICallback
        {
            std::function<void(T)> func;
            std::promise<void> promise;
            T data;
            FunctionCallback(const T& d, const std::function<void(T)> f) : func(f), data(d) {}
            virtual ~FunctionCallback() {}
            virtual void run();
        };
        template<typename _return_type> 
        struct EAGLE_EXPORTS LambdaCallback: public ICallback
        {
            std::function<_return_type()> func;
            std::promise<_return_type> promise;
            
            LambdaCallback(const std::function<_return_type()>& f) : func(f) {}
            ~LambdaCallback() {}
            virtual void run();
        };
        template<typename _return_type>
        struct EAGLE_EXPORTS LambdaCallbackEvent: public ICallbackEventLoop
        {
            std::function<_return_type()> func;
            std::promise<_return_type> promise;

            LambdaCallbackEvent(const std::function<_return_type()>& f) : func(f) {}
            ~LambdaCallbackEvent() {}
            virtual void run();
        };

        template<> struct EAGLE_EXPORTS LambdaCallback<void> : public ICallback
        {
            std::function<void()> func;
            std::promise<void> promise;

            LambdaCallback(const std::function<void()>& f);
            ~LambdaCallback();
            virtual void run();
        };
        template<> struct EAGLE_EXPORTS LambdaCallbackEvent<void> : public ICallbackEventLoop
        {
            std::function<void()> func;
            std::promise<void> promise;

            LambdaCallbackEvent(const std::function<void()>& f);
            ~LambdaCallbackEvent();
            virtual void run();
        };
        // Lambda functions
        template<typename _Ty> auto
            enqueue_callback_async(_Ty function, cv::cuda::Stream& stream)->std::future<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>
        {
            auto fc = new LambdaCallback<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>(function);
            auto future = fc->promise.get_future();
            stream.enqueueHostCallback(&ICallback::cb_func_async, fc);
            return future;
        }
        template<typename _Ty> auto
            enqueue_callback(_Ty function, cv::cuda::Stream& stream)->std::future<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>
        {
            auto fc = new LambdaCallback<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>(function);
            auto future = fc->promise.get_future();
            stream.enqueueHostCallback(&ICallback::cb_func, fc);
            return future;
        }

        template<typename _Ty> 
        auto enqueue_callback_async(_Ty function, size_t thread, cv::cuda::Stream& stream)->std::future<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>
        {
            auto fc = new LambdaCallbackEvent<typename pplx::details::_TaskTypeFromParam<_Ty>::_Type>(function);
            fc->event_loop_thread_id = thread;
            auto future = fc->promise.get_future();
            stream.enqueueHostCallback(&ICallback::cb_func_async_event_loop, fc);
            return future;
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
        template<typename T> void LambdaCallbackEvent<T>::run()
        {
            promise.set_value(func());
        }
        template<typename T, typename R> void FunctionCallback<T, R>::run()
        {
            promise.set_value(func(data));
        }
        template<typename T> void FunctionCallback<T, void>::run()
        {
            func(data);
            promise.set_value();
        }
        template<typename T> void Callback<T>::run()
        {

        }
    }
}



