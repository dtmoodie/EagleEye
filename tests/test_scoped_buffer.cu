#include <EagleLib/utilities/GpuMatAllocators.h>
#include <EagleLib/utilities/BufferPool.hpp>
#include <EagleLib/Thrust_interop.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudafilters.hpp>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

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

int main()
{
	cv::cuda::Stream streams[2];
	auto filter = cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 1);
	cv::cuda::GpuMat src(10000,1000,CV_32F);

	auto srcBegin = GpuMatBeginItr<float>(src,0);
	
	thrust::transform(
		thrust::make_counting_iterator(0), 
		thrust::make_counting_iterator(src.cols*src.rows), 
		srcBegin, prg(-1, 1));

	EagleLib::DelayedDeallocator* alloc = new EagleLib::DelayedDeallocator();
	
	alloc->deallocateDelay = 1000;
	cv::cuda::GpuMat::setDefaultAllocator(alloc);
	for (int i = 0; i < 1000; ++i)
	{
		EagleLib::scoped_buffer scoped_buf(streams[i % 2]);
		filter->apply(src, scoped_buf.GetMat(), streams[i % 2]);
		EagleLib::scoped_buffer::GarbageCollector::Run();
	}
	return 0;
}