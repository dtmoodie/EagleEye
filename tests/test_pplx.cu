#define CPPREST_FORCE_PPLX 1
#include "pplx/pplx.h"
#include "pplx/pplxtasks.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/highgui.hpp>

#include "Thrust_interop.hpp"

#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
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
	cv::cuda::GpuMat bigTestMat;
	bigTestMat.create(10000, 10000, CV_32F);

	auto valueBegin = GpuMatBeginItr<float>(bigTestMat);
	auto valueEnd = GpuMatEndItr<float>(bigTestMat);

	thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(bigTestMat.size().area()), valueBegin, prg(-1, 1));

	cv::cuda::HostMem* user_data = new cv::cuda::HostMem();
	bigTestMat.download(*user_data, stream);
	std::cout << "Enqueuing data on thread " << boost::this_thread::get_id() << " at time " << clock() << std::endl;
	stream.enqueueHostCallback(cuda_callback<cv::cuda::HostMem, disp_operator>, user_data);


	while (1)
	{

	}
	return 0;
}