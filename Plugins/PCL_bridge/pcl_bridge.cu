#include "pcl_bridge.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

void __global__ centroid_kernel(const cv::cuda::PtrStepSz<float> points, cv::cuda::PtrStepSz<float> moments, cv::cuda::PtrStepSz<int> mask, int mask_value)
{
	__shared__ float smem[3];
	if (threadIdx.x == 0)
	{
		smem[0] = 0;
		smem[1] = 0;
		smem[2] = 0;
	}
	__syncthreads();
	if (mask.data)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < points.rows;
			i += blockDim.x * gridDim.x)
		{
			if (mask.data[i] == mask_value)
			{
				atomicAdd(smem  , points.ptr(i)[0]);
				atomicAdd(smem+1, points.ptr(i)[1]);
				atomicAdd(smem+2, points.ptr(i)[2]);
			}
		}
	} 
	else
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < points.rows;
			i += blockDim.x * gridDim.x)
		{
			atomicAdd(smem    , points.ptr(i)[0]);
			atomicAdd(smem + 1, points.ptr(i)[1]);
			atomicAdd(smem + 2, points.ptr(i)[2]);
		}
	}
	__syncthreads();
	if (threadIdx.x == 0)
	{
		moments.data[0] = smem[0];
		moments.data[1] = smem[1];
		moments.data[2] = smem[2];
	}
}



// Moments are arranged as follows:
// Cx, Cy, Cz, U200, U020, U002

void calculateHuMoment(cv::cuda::GpuMat input, cv::cuda::GpuMat& output, cv::cuda::GpuMat mask, int mask_value, cv::cuda::Stream stream)
{
	CV_Assert(input.depth() == CV_32F && "Only accepts FP32 point clouds");
	if (input.channels() == 3 || input.channels() == 4)
	{
		input = input.reshape(1, input.rows * input.cols); // Reshape a 2d image into tensor format
	}
	if (!mask.empty())
	{
		CV_Assert(mask.size().area() == input.rows && "Mask not equivalent size to the input point cloud");
		if (mask.cols > 1)
			mask = mask.reshape(1, input.rows * input.cols);
	}
	int numSMs;
	int devId;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);


	if (output.cols != 6 || output.depth() != CV_32F)
	{
		output = cv::cuda::createContinuous(1, 6, CV_32F);
	}
	centroid_kernel <<<numSMs * 32, 256, 3 * sizeof(float), cv::cuda::StreamAccessor::getStream(stream) >> >(cv::cuda::PtrStepSz<float>(input), cv::cuda::PtrStepSz<float>(output), cv::cuda::PtrStepSz<int>(mask), mask_value);
}