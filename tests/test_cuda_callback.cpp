#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "Aquila/utilities/cuda/CudaCallbacks.hpp"

int main()
{
    cv::cuda::Stream stream;
    cv::cuda::GpuMat mat(10000, 10000, CV_32F);
    //mat.setTo(cv::Scalar())

    return 0;
}
