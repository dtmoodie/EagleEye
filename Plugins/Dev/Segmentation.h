#include "nodes/Node.h"
#include "CudaUtils.hpp"
#ifdef __cplusplus
extern "C"{
#endif
    IPerModuleInterface* GetModule();

#ifdef __cplusplus
}
#endif

namespace EagleLib
{
    class OtsuThreshold: public Node
    {
    public:
        OtsuThreshold();
        void Init(bool firstInit);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentWatershed: public Node
    {
    public:
        SegmentWatershed();
        void Init(bool firstInit);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentGrabCut: public Node
    {
        cv::Mat bgdModel;
        cv::Mat fgdModel;
        ConstBuffer<cv::Mat> maskBuf;
    public:
        SegmentGrabCut();
        void Init(bool firstInit);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentKMeans: public Node
    {
    public:
        SegmentKMeans();
        void Init(bool firstInit);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    class ManualMask: public Node
    {
        enum MaskType
        {
            Circular = 0,
            Rectangular = 1
        };
        cv::cuda::GpuMat mask;
    public:
        ManualMask();
        void Init(bool firstInit);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
