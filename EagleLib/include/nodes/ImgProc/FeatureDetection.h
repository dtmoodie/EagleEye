#include <nodes/Node.h>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace EagleLib
{
    class CV_EXPORTS GoodFeaturesToTrackDetector : public Node
    {
    public:
        GoodFeaturesToTrackDetector();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };
}
