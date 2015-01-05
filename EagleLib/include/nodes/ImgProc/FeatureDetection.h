#include <nodes/Node.h>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace EagleLib
{

    class GoodFeaturesToTrackDetector: public Node
    {
    public:
        GoodFeaturesToTrackDetector();
        GoodFeaturesToTrackDetector(bool drawResults_);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);

    private:
        int imgType;

    };
}
