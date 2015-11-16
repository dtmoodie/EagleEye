#pragma once
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include "EagleLib/utilities/CudaUtils.hpp"

void filterPointCloud(cv::cuda::GpuMat inputPointCloud, cv::cuda::GpuMat& outputPointCloud,
	cv::cuda::GpuMat mask, cv::cuda::GpuMat& resultSize, int flagValue, cv::cuda::Stream stream);