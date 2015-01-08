#include "nodes/Display/ImageDisplay.h"

using namespace EagleLib;

ImageDisplay::ImageDisplay()
{

}

ImageDisplay::ImageDisplay(boost::function<void(cv::Mat)> cpuCallback_)
{
    cpuDisplayCallback = cpuCallback_;
}
ImageDisplay::ImageDisplay(boost::function<void (cv::cuda::GpuMat)> gpuCallback_)
{
    gpuDisplayCallback = gpuCallback_;
}

cv::cuda::GpuMat
ImageDisplay::doProcess(cv::cuda::GpuMat& img)
{
    cv::Mat h_img(img);
    cv::imshow("test",h_img);
    return img;
}

