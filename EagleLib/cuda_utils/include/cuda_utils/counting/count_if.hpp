#pragma once

namespace cu
{
	void count_if_greater(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& lower_bound, cv::cuda::Stream& stream);
	void count_if_not(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& not_equal, cv::cuda::Stream& stream);
	void count_if_less(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& upper_bound, cv::cuda::Stream& stream);
}