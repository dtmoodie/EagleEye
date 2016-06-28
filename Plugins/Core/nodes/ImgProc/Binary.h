#include "EagleLib/nodes/Node.h"
#include "EagleLib/Defs.hpp"
#include <EagleLib/Project_defs.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include <EagleLib/ParameteredIObjectImpl.hpp>
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

SETUP_PROJECT_DEF

namespace EagleLib
{
    namespace Nodes
    {
    
    class MorphologyFilter: public Node
    {
    public:
        MorphologyFilter();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);

    };

    class FindContours: public Node
    {
    public:
        FindContours();
        virtual void NodeInit(bool firstInit);
        //virtual void findContours(cv::cuda::HostMem img);
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
    };

    class PruneContours: public Node
    {
    public:
        BEGIN_PARAMS(PruneContours)
            PARAM(int, min_area, 20)
            PARAM(int, max_area, 500)
        END_PARAMS;
        PruneContours();

        virtual void NodeInit(bool firstInit);
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
    };

    class ContourBoundingBox: public Node
    {
    public:
        ContourBoundingBox();
        virtual void NodeInit(bool firstInit);
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
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
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
        void runFilter();
    };

    class DrawContours: public Node
    {
    public:
        BEGIN_PARAMS(DrawContours)
            PARAM(cv::Scalar, draw_color, cv::Scalar(0,0,255))
            PARAM(int, draw_thickness, 8);
        END_PARAMS;

        DrawContours();
        virtual void NodeInit(bool firstInit);
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
    };

    class DrawRects: public Node
    {
    public:
        DrawRects();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    }
}
