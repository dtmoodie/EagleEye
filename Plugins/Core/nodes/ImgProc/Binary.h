#include "nodes/Node.h"
#include "EagleLib/Defs.hpp"

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

SETUP_PROJECT_DEF

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
		virtual void findContours(cv::cuda::HostMem img);
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
