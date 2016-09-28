






#include "src/precompile.hpp"

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
        MO_DERIVE(MorphologyFilter, Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output, SyncedMemory());
            ENUM_PARAM(mo::EnumParameter, structuring_element_type, cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE);
            ENUM_PARAM(mo::EnumParameter, morphology_type, cv::MORPH_ERODE, cv::MORPH_DILATE, cv::MORPH_OPEN, cv::MORPH_CLOSE, cv::MORPH_GRADIENT, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT);
            PARAM(int, iterations, 1);
            PARAM(cv::Mat, structuring_element, cv::getStructuringElement(0, cv::Size(5,5)));
            PARAM(cv::Point, anchor_point, cv::Point(-1,-1));
            PARAM(int, structuring_element_size, 5);
        MO_END;
        
    protected:
        bool ProcessImpl();
        cv::Ptr<cv::cuda::Filter> filter;
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
        MO_DERIVE(PruneContours, Node)
            PARAM(int, min_area, 20)
            PARAM(int, max_area, 500)
        MO_END;
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
        MO_DERIVE(DrawContours, Node)
            PARAM(cv::Scalar, draw_color, cv::Scalar(0,0,255))
            PARAM(int, draw_thickness, 8);
        MO_END;

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
