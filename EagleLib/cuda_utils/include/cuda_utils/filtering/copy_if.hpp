#pragma once

#include "cuda_utils/export.hpp"
#include <opencv2/core/cuda.hpp>

namespace cu
{
	// Given an input 1 to 4 channel mat, copy all values into a linear vector output
	// if output's size is already defined, will only copy up to that number of values
	// If output's size is not defined, will count the number of values that satisfy the requirement, and then copy those values
	void copy_if_greater(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& lower_bound, cv::cuda::Stream& stream);
	void copy_if_not(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& not_equal, cv::cuda::Stream& stream);
	void copy_if_equal(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& not_equal, cv::cuda::Stream& stream);
	void copy_if_less(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& upper_bound, cv::cuda::Stream& stream);
}