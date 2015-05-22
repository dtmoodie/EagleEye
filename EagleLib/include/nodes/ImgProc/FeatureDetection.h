#include <nodes/Node.h>
#include "CudaUtils.hpp"

namespace EagleLib
{
    class GoodFeaturesToTrackDetector : public Node
    {
        ConstBuffer<cv::cuda::GpuMat> greyImgs;
        ConstBuffer<std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> detectedPoints;
        cv::cuda::GpuMat detectedCorners;
        virtual void detect(cv::cuda::GpuMat img, cv::cuda::GpuMat mask,
                    cv::cuda::GpuMat& keyPoints,
                    cv::cuda::GpuMat& descriptors,
                    cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    public:
        GoodFeaturesToTrackDetector();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class FastFeatureDetector: public Node
    {
        ConstBuffer<std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> detectedPoints;
        virtual void detect(cv::cuda::GpuMat img, cv::cuda::GpuMat mask,
                    cv::cuda::GpuMat& keyPoints,
                    cv::cuda::GpuMat& descriptors,
                    cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    public:
        FastFeatureDetector();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class ORBFeatureDetector: public Node
    {
        ConstBuffer<std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> detectedPoints;
        virtual void detect(cv::cuda::GpuMat img, cv::cuda::GpuMat mask,
                    cv::cuda::GpuMat& keyPoints,
                    cv::cuda::GpuMat& descriptors,
                    cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    public:
        ORBFeatureDetector();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };


    class HistogramRange: public Node
    {
        cv::cuda::GpuMat levels;
        void updateLevels();
    public:
        HistogramRange();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
