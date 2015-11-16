#include "nodes/Node.h"
#include "opencv2/cudastereo.hpp"
#include "EagleLib/utilities/CudaUtils.hpp""
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
        cv::Ptr<cv::cuda::StereoBeliefPropagation> bp;
    public:
        StereoBeliefPropagation();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class StereoConstantSpaceBP: public Node
    {
        cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp;
    public:
        StereoConstantSpaceBP();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    class UndistortStereo: public Node
    {
        cv::cuda::GpuMat mapY, mapX;
        cv::cuda::HostMem X, Y;

    public:
        UndistortStereo();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
