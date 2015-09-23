#include "flann.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thrust/transform.h>
#include "Thrust_interop.hpp"
#include "thrust/system/cuda/execution_policy.h"

template<typename T1, typename T2>
__global__ void filterPointCloud(cv::cuda::PtrStepSz<T1> inputPointCloud,
								 cv::cuda::PtrStepSz<T1> outputPointCloud,
								 T2* pMask,
								 T2* resultSize,
								 T2 flag)
{
	//__shared__ T2 insertionItr;
	T2 tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < inputPointCloud.rows;
		i += blockDim.x * gridDim.x)
	{
		if (pMask[i] == flag)
		{
			T2 insertion = atomicAdd(resultSize, 1);
			auto dest = outputPointCloud.ptr(insertion);
			auto src = inputPointCloud.ptr(i);
			dest[0] = src[0];
			dest[1] = src[1];
			dest[2] = src[2];
			dest[3] = src[3];
		}
	}
}


void filterPointCloud(cv::cuda::GpuMat inputPointCloud, cv::cuda::GpuMat& outputPointCloud, cv::cuda::GpuMat mask, cv::cuda::GpuMat& resultSize, int flagValue, cv::cuda::Stream stream)
{
	CV_Assert(inputPointCloud.type() == CV_32F &&
			  inputPointCloud.channels() == 1 && 
			 (inputPointCloud.cols == 4 || 
			  inputPointCloud.cols == 3 && inputPointCloud.step == 16) && 
			  "Only accepts tensor format with XYZ1");
	CV_Assert(mask.isContinuous());

	outputPointCloud.create(inputPointCloud.size(), inputPointCloud.type());
	resultSize.create(1, 1, CV_32S);
	resultSize.setTo(cv::Scalar(0), stream);

	int numSMs;
	int devId;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
	filterPointCloud<float, int> <<<numSMs*32, 256, 0, cv::cuda::StreamAccessor::getStream(stream) >>>(
			cv::cuda::PtrStepSz<float>(inputPointCloud),
			cv::cuda::PtrStepSz<float>(outputPointCloud),
			(int*)mask.data,
			(int*)resultSize.data,
			flagValue);
#if _DEBUG
	//stream.waitForCompletion();
#endif
}

