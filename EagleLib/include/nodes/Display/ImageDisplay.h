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
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        void displayImage(cv::cuda::HostMem image);

    };
    class OGLImageDisplay: public Node
    {
        std::string prevName;

    public:
        OGLImageDisplay();

        OGLImageDisplay(boost::function<void(cv::cuda::GpuMat, Node*)> gpuCallback_);
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    };
    class KeyPointDisplay: public Node
    {
        ConstEventBuffer<std::pair<cv::cuda::HostMem, cv::cuda::HostMem>> hostData;
        int displayType;
    public:

        KeyPointDisplay();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        cv::Mat uicallback();
        virtual void Serialize(ISimpleSerializer *pSerializer);
    };
    class FlowVectorDisplay: public Node
    {
        // First buffer is the image, second is a pair of the points to be used
        ConstEventBuffer<cv::cuda::HostMem[4]> hostData;
    public:
        FlowVectorDisplay();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        cv::Mat uicallback();
        virtual void Serialize(ISimpleSerializer *pSerializer);

    };
}
