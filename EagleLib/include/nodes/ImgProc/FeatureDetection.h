#include <nodes/Node.h>


namespace EagleLib
{
    class GoodFeaturesToTrackDetector : public Node
    {
        cv::cuda::GpuMat greyImg;
        cv::cuda::GpuMat detectedCorners;
        virtual cv::cuda::GpuMat detect(cv::cuda::GpuMat img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    public:
        GoodFeaturesToTrackDetector();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
