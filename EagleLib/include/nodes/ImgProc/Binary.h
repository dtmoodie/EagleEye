#include "nodes/Node.h"
namespace EagleLib
{
    class MorphologyFilter: public Node
    {
    public:
        MorphologyFilter();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);

    };

    class FindContours: public Node
    {
    public:
        FindContours();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };

    class ContourBoundingBox: public Node
    {
    public:
        ContourBoundingBox();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
}
