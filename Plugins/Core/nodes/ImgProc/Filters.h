#include "EagleLib/nodes/Node.h"
#include <EagleLib/rcc/external_includes/cv_cudafeatures3d.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class Sobel: public Node
    {
    public:
        Sobel();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);

    };

    class Canny: public Node
    {
		cv::Ptr<cv::cuda::CannyEdgeDetector> detector;
    public:
        Canny();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };

    class Laplacian: public Node
    {
    public:
        Laplacian();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class BiLateral: public Node
    {
    public:
        BiLateral();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class MeanShiftFilter: public Node
    {
    public:
        MeanShiftFilter();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class MeanShiftProc: public Node
    {
    public:
        MeanShiftProc();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class MeanShiftSegmentation: public Node
    {
    public:
        MeanShiftSegmentation();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    }
}
