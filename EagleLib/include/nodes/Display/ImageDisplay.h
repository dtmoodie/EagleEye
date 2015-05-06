#include "nodes/Node.h"

#include <opencv2/highgui.hpp>
#include <CudaUtils.hpp>
namespace EagleLib
{
    class QtImageDisplay: public Node
    {
        std::string prevName;
        cv::cuda::HostMem hostImage;
    public:
        QtImageDisplay();
        QtImageDisplay(boost::function<void(cv::Mat, Node*)> cpuCallback_);
        QtImageDisplay(boost::function<void(cv::cuda::GpuMat, Node*)> gpuCallback_);
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
        void displayImage(cv::cuda::HostMem image);

    };
    class OGLImageDisplay: public Node
    {
        std::string prevName;

    public:
        OGLImageDisplay();

        OGLImageDisplay(boost::function<void(cv::cuda::GpuMat, Node*)> gpuCallback_);
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());

    };
    class KeyPointDisplay: public Node
    {
        ConstBuffer<cv::cuda::HostMem> keyPointMats;
        ConstBuffer<cv::cuda::HostMem> hostImages;
        int displayType;
    public:

        KeyPointDisplay();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
        cv::Mat uicallback();
    };
}
