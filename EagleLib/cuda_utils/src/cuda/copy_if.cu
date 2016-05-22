#include <opencv2/core/cuda/vec_traits.hpp>
#include <cuda.h>

using namespace cv;
using namespace cv::cuda;

namespace cu
{
	namespace copy_if
	{
		template<typename T, int N> struct caller
		{
			static void greater(const PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream)
			{
			
			}
			static void if_not(const PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream)
			{
			
			}
			static void if_equal(const PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream)
			{
			
			}
			static void less(const PtrStepSz<typename device::TypeVec<T, N>::vec_type> input, PtrStepSz<typename device::TypeVec<T, N>::vec_type> output, const cv::Scalar& lower_bound, cudaStream_t stream)
			{
			
			}
		};
	}
}