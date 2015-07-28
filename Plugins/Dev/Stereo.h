#include "nodes/Node.h"
#include "opencv2/cudastereo.hpp"
#include "CudaUtils.hpp"
namespace EagleLib
{
    class StereoBM: public Node
    {
        cv::Ptr<cv::cuda::StereoBM> stereoBM;
        BufferPool<cv::cuda::GpuMat, EventPolicy> disparityBuf;
    public:
        StereoBM();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    class StereoBilateralFilter: public Node
    {
    public:
        StereoBilateralFilter();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class StereoBeliefPropagation: public Node
    {
    public:
        StereoBeliefPropagation();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class StereoConstantSpaceBP: public Node
    {
    public:
        StereoConstantSpaceBP();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    class UndistortStereo: public Node
    {
        cv::cuda::GpuMat mapY, mapX;

    public:
        UndistortStereo();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
