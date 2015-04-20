#include "nodes/Utility/Frame.h"

using namespace EagleLib;
void FrameRate::Init(bool firstInit)
{

}

cv::cuda::GpuMat FrameRate::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    updateParameter<double>("Framerate", 1000.0 / delta.total_milliseconds());
    return img;
}
