#include "nodes/Display/ImageDisplay.h"
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;

ImageDisplay::ImageDisplay()
{

}

ImageDisplay::ImageDisplay(boost::function<void(cv::Mat)> cpuCallback_)
{
    cpuCallback = cpuCallback_;
}
ImageDisplay::ImageDisplay(boost::function<void (cv::cuda::GpuMat)> gpuCallback_)
{
    gpuCallback = gpuCallback_;
}

cv::cuda::GpuMat
ImageDisplay::doProcess(cv::cuda::GpuMat& img)
{
    cv::Mat h_img(img);
    cv::imshow("test",h_img);
    return img;
}

