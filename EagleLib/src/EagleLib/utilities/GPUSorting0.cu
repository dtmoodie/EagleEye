#include <EagleLib/Thrust_interop.hpp>
#include "EagleLib/utilities/GPUSortingPriv.hpp"
namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template EAGLE_EXPORTS void sortAscending<uchar>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescending<uchar>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortAscendingEachRow<uchar>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescendingEachRow<uchar>(cv::cuda::GpuMat&, cudaStream_t);

            template EAGLE_EXPORTS void sortAscending<char>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescending<char>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortAscendingEachRow<char>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescendingEachRow<char>(cv::cuda::GpuMat&, cudaStream_t);
        }
    }
}

