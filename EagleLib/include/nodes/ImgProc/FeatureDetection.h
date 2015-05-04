#include <nodes/Node.h>


namespace EagleLib
{
    class GoodFeaturesToTrackDetector : public Node
    {
    public:
        GoodFeaturesToTrackDetector();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };
}
