#pragma once
#include <EagleLib/Detail/Export.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime_api.h>
namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template<typename T> EAGLE_EXPORTS void sortAscending(cv::cuda::GpuMat& out, cudaStream_t stream);
            template<typename T> EAGLE_EXPORTS void sortDescending(cv::cuda::GpuMat& out, cudaStream_t stream);
            template<typename T> EAGLE_EXPORTS void sortAscendingEachCol(cv::cuda::GpuMat& out, cudaStream_t stream);
            template<typename T> EAGLE_EXPORTS void sortDescendingEachCol(cv::cuda::GpuMat& out, cudaStream_t stream);
            template<typename T> EAGLE_EXPORTS void sortAscendingEachRow(cv::cuda::GpuMat& out, cudaStream_t stream);
            template<typename T> EAGLE_EXPORTS void sortDescendingEachRow(cv::cuda::GpuMat& out, cudaStream_t stream);
        }
        void EAGLE_EXPORTS sort(InputArray src, OutputArray dst, int flags, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    }
}