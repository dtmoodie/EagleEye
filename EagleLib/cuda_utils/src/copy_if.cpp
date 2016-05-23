#include "cuda_utils/filtering/copy_if.hpp"
#include <opencv2/cudev.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
using namespace cv;
using namespace cv::cuda;


namespace copy_if
{
	template<typename T, int N> struct caller
	{
		static void greater(PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream);
		static void if_not(PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream);
		static void if_equal(PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream);
		static void less(PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream);
	};
}


void copy_if_greater(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& lower_bound, cv::cuda::Stream& stream)
{

}
void copy_if_not(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& not_equal, cv::cuda::Stream& stream)
{

}
void copy_if_equal(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& not_equal, cv::cuda::Stream& stream)
{

}
void copy_if_less(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::Scalar& upper_bound, cv::cuda::Stream& stream)
{

}