#pragma once
#include <opencv2/core/cuda.hpp>

void calculateHuMoment(cv::cuda::GpuMat input, cv::cuda::GpuMat& output, cv::cuda::GpuMat mask = cv::cuda::GpuMat(), int mask_value = -1, cv::cuda::Stream stream = cv::cuda::Stream::Null());