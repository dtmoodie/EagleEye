#pragma once
#include <opencv2/core/cuda.hpp>
#include <pplx/pplxtasks.h>

namespace EagleLib
{
	namespace cuda
	{
		struct ICallback
		{
			static void cb_func(int status, void* user_data);
			virtual void run() = 0;
		};
		
		template<typename T, typename C> void enqueue_callback(const T& user_data, cv::cuda::Stream& stream)
		{
			static_assert(std::is_base_of<ICallback, C>::value);
			stream.enqueueHostCallback(ICallback::cb_func, new C(T));
		}

		template<typename T> class Callback: public ICallback
		{
			Callback(const T& data);
			virtual void run();
		};


		// Implementations

		template<typename T> void Callback<T>::run()
		{

		}
	}
}



