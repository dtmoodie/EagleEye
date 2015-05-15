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

    class HistogramThreshold: public Node
    {
        cv::cuda::GpuMat* inputHistogram;
        cv::cuda::GpuMat* inputImage;
        cv::cuda::GpuMat* inputMask;
        cv::cuda::Stream _stream;
        cv::cuda::GpuMat lowerMask;
        cv::cuda::GpuMat upperMask;
        enum ThresholdType
        {
            KeepCenter = 0,
            SuppressCenter
        };
        ThresholdType type;
    public:

        HistogramThreshold();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
        void runFilter();
    };
}
