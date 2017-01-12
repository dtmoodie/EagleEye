#include <EagleLib/Thrust_interop.hpp>
#include "EagleLib/utilities/GPUSortingPriv.hpp"
namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template EAGLE_EXPORTS void sortAscending<float>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescending<float>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortAscendingEachRow<float>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescendingEachRow<float>(cv::cuda::GpuMat&, cudaStream_t);

            template EAGLE_EXPORTS void sortAscending<double>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescending<double>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortAscendingEachRow<double>(cv::cuda::GpuMat&, cudaStream_t);
            template EAGLE_EXPORTS void sortDescendingEachRow<double>(cv::cuda::GpuMat&, cudaStream_t);
        }
    }
}
