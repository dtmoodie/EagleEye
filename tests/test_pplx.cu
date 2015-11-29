#define CPPREST_FORCE_PPLX 1
#include "pplx/pplx.h"
#include "pplx/pplxtasks.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/highgui.hpp>

#include "Thrust_interop.hpp"
#include "EagleLib/utilities/CudaCallbacks.hpp"
#include "EagleLib/utilities/CpuMatAllocators.h"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/log/trivial.hpp>
#include <time.h>
struct prg
{
	float a, b;

	__host__ __device__
		prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

	__host__ __device__
		float operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(a, b);
		rng.discard(n);
		return dist(rng);
	}
};

template<typename T> struct disp_operator
{
	static void process(T* data)
	{
		std::cout << "Received data on thread " << boost::this_thread::get_id() << " at time " << clock() << std::endl;
		
		pplx::create_task([data] 
		{
			auto start = clock();
			std::cout << "Rendering data on thread " << boost::this_thread::get_id() << " at time " << start << std::endl;
			cv::imshow("Data", *data);
			cv::waitKey(1);
			std::cout << "Displaying data took "  << clock() - start << " ms\n";
		});
	}
};


template<typename T, template<typename> class OP> void cuda_callback(int status, void* user_data)
{
	OP<T>::process(static_cast<T*>(user_data));
}

int main()
{
	cv::cuda::Stream stream;
	cv::cuda::Stream downloadStream;
	cv::cuda::Event downloadReadyEvent;
	cv::cuda::GpuMat bigTestMat[2];
	cv::Mat::setDefaultAllocator(EagleLib::CpuPinnedAllocator::instance());
	bigTestMat[0].create(10000, 10000, CV_32F);
	bigTestMat[1].create(10000, 10000, CV_32F);

	

	
	auto start = clock();
    for (int i = 0; i < 50; ++i)
    {
        cv::Mat user_data;
        {
			auto valueBegin = GpuMatBeginItr<float>(bigTestMat[i % 2]);
			auto valueEnd = GpuMatEndItr<float>(bigTestMat[i%2]);
            EagleLib::cuda::scoped_event_stream_timer timer(stream, "Transform time");

            thrust::transform(thrust::system::cuda::par.on(cv::cuda::StreamAccessor::getStream(stream)), 
				thrust::make_counting_iterator(0), 
				thrust::make_counting_iterator(bigTestMat[i%2].size().area()), 
				valueBegin, 
				prg(-1, 1));

			downloadReadyEvent.record(stream);
        }
        
        
        {
            EagleLib::cuda::scoped_event_stream_timer timer(downloadStream, "Download time");
			downloadStream.waitEvent(downloadReadyEvent);
            bigTestMat[i%2].download(user_data, downloadStream);
        }
        
        {
            EagleLib::cuda::scoped_event_stream_timer timer(downloadStream, "Display callback");
            /*EagleLib::cuda::enqueue_callback_async<cv::Mat, void>(
				user_data,
                [](cv::Mat img)->void
            {
                //cv::imshow("Display", img);
            }, downloadStream);*/

			EagleLib::cuda::enqueue_callback_async(
			[user_data]()->void
			{

			}, downloadStream);
        }
		std::cout << "Loop time " << clock() - start << " ms\n";
		start = clock();
    }
	return 0;
}