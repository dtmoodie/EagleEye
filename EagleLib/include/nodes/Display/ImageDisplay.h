#include "nodes/Node.h"

#include <opencv2/highgui.hpp>

namespace EagleLib
{
    class CV_EXPORTS ImageDisplay: public Node
    {
    public:
        ImageDisplay();
        ImageDisplay(boost::function<void(cv::Mat)> cpuCallback_);
        ImageDisplay(boost::function<void(cv::cuda::GpuMat)> gpuCallback_);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
        

    };
}
