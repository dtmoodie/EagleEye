#pragma once
#include <opencv2/core.hpp>

namespace cv
{
namespace cuda
{
    void rectangle(cv::cuda::GpuMat& img, const cv::Rect& rect, cv::Scalar color, int thickness, cv::cuda::Stream& stream);
}
}
