#include "nodes/Node.h"

#include <opencv2/highgui.hpp>

namespace EagleLib
{
    class CV_EXPORTS QtImageDisplay: public Node
    {
        std::string prevName;
    public:
        QtImageDisplay();
        QtImageDisplay(boost::function<void(cv::Mat)> cpuCallback_);
        QtImageDisplay(boost::function<void(cv::cuda::GpuMat)> gpuCallback_);
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());

    };
    class CV_EXPORTS OGLImageDisplay: public Node
    {
        std::string prevName;
    public:
        OGLImageDisplay();

        OGLImageDisplay(boost::function<void(cv::cuda::GpuMat)> gpuCallback_);
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());

    };
}
