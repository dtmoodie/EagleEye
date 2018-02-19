#pragma once
#include <opencv2/core/cuda.hpp>

void Depth2XYZ(cv::cuda::GpuMat depth, cv::cuda::GpuMat& XYZ, cv::cuda::Stream stream);
