#include "freenect.cuh"

#include <cuda.h>
#include <cuda_runtime.h>



#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudev.hpp>

__global__ void depth2XYZ_kernel(cv::cuda::PtrStepSz<unsigned short> depth, cv::cuda::PtrStepSz<cv::Vec4f> XYZ)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x_ = 320 - x;
	const int y_ = 240 - y;
	const float f = 570.3;
	const float depth_ = depth.ptr(y)[x];
	XYZ.ptr(y)[x].val[0] = x_ * depth_ / f;
	XYZ.ptr(y)[x].val[1] = y_ * depth_ / f;
	XYZ.ptr(y)[x].val[2] = depth_;
	XYZ.ptr(y)[x].val[3] = 0;

}



void Depth2XYZ(cv::cuda::GpuMat depth, cv::cuda::GpuMat& XYZ, cv::cuda::Stream stream)
{
	CV_Assert(!depth.empty());
	CV_Assert(depth.type() == CV_16UC1);

	XYZ.create(depth.size(), CV_32FC4);

	dim3 threadsPerBlock(16, 16);
	dim3 blocks(cv::cudev::divUp(depth.cols, 16), cv::cudev::divUp(depth.rows, 16));
	
	depth2XYZ_kernel << <blocks, threadsPerBlock, 0, cv::cuda::StreamAccessor::getStream(stream) >> >(depth, XYZ);
	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	if (stream == NULL)
	{
		cudaDeviceSynchronize();
	}

}