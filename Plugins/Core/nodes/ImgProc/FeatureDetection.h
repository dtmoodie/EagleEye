#include <nodes/Node.h>
#include <external_includes/cv_cudafeatures2d.hpp>
#include "EagleLib/utilities/CudaUtils.hpp""
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
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
		cv::Ptr<cv::cuda::Feature2DAsync> detector;
        
    public:
        FastFeatureDetector();
        virtual void Init(bool firstInit);
		virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class ORBFeatureDetector: public Node
    {
		cv::Ptr<cv::cuda::ORB> detector;
		
        
    public:
        ORBFeatureDetector();
        virtual void Init(bool firstInit);
		virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };


    class HistogramRange: public Node
    {
        cv::cuda::GpuMat levels;
        void updateLevels(int type);
    public:
        HistogramRange();
        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
