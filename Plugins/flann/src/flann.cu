#include "flann.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

template<typename T1, typename T2>
__global__ void filterPointCloud(cv::cuda::PtrStepSz<T1> inputPointCloud,
								 cv::cuda::PtrStepSz<T1> outputPointCloud,
								 T2* pMask,
								 T2* resultSize,
								 T2 step,
								 T2 flag)
{
	__shared__ T2 insertionItr;
	T2 tid = threadIdx.x;
	if (tid == 0)
		insertionItr = 0;
	__syncthreads();

	for (T2 i = 0; i < inputPointCloud.rows; i += step)
	{
		if (pMask[i] == flag)
		{
			T2 insertion = atomicAdd(&insertionItr, 1);
			auto dest = outputPointCloud.ptr(insertion);
			auto src = inputPointCloud.ptr(i);
			dest[0] = src[0];
			dest[1] = src[1];
			dest[2] = src[2];
			dest[3] = src[3];
		}
	}
	__syncthreads();
	if (tid == 0)
		*resultSize = insertionItr;
}


void filterPointCloud(cv::cuda::GpuMat inputPointCloud, EagleLib::GpuResized<cv::cuda::GpuMat>& outputPointCloud, cv::cuda::GpuMat mask, cv::cuda::GpuMat& resultSize, int flagValue, cv::cuda::Stream stream)
{
	CV_Assert(inputPointCloud.type() == CV_32F &&
			  inputPointCloud.channels() == 1 && 
			 (inputPointCloud.cols == 4 || 
			  inputPointCloud.cols == 3 && inputPointCloud.step == 16) && 
			  "Only accepts tensor format with XYZ1");
	CV_Assert(mask.isContinuous());

	outputPointCloud.data.create(inputPointCloud.size(), inputPointCloud.type());
	resultSize.create(1, 1, CV_32S);

	int step = inputPointCloud.rows / 1024;
	
	filterPointCloud<float, int> <<<1, 1024, sizeof(int), cv::cuda::StreamAccessor::getStream(stream) >>>(
			cv::cuda::PtrStepSz<float>(inputPointCloud),
			cv::cuda::PtrStepSz<float>(outputPointCloud.data),
			(int*)mask.data,
			(int*)outputPointCloud.GpuSetSize,
			step, flagValue);

}