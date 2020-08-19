#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

int main()
{
    cv::cuda::Stream stream;
    cv::cuda::GpuMat mat(10000, 10000, CV_32F);

    return 0;
}
