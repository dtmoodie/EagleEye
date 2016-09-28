#include "EagleLib/nodes/Node.h"
#include "EagleLib/Detail/Export.hpp"
#include "EagleLib/Detail/PluginExport.hpp"
#include <EagleLib/rcc/external_includes/cv_cudafilters.hpp>

#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <MetaObject/Parameters/TypedInputParameter.hpp>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"

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
        static ::mo::EnumParameter StructuringTypes()
        {
            ::mo::EnumParameter ret;
            ret.addEnum(ENUM(cv::MORPH_RECT));
            ret.addEnum(ENUM(cv::MORPH_CROSS));
            ret.addEnum(ENUM(cv::MORPH_ELLIPSE));
            return ret;
        }
        static ::mo::EnumParameter MorphTypes()
        {
            ::mo::EnumParameter ret;
            ret.addEnum(ENUM(cv::MORPH_ERODE));
            ret.addEnum(ENUM(cv::MORPH_DILATE));
            ret.addEnum(ENUM(cv::MORPH_OPEN));
            ret.addEnum(ENUM(cv::MORPH_CLOSE));
            ret.addEnum(ENUM(cv::MORPH_GRADIENT));
            ret.addEnum(ENUM(cv::MORPH_TOPHAT));
            ret.addEnum(ENUM(cv::MORPH_BLACKHAT));
            return ret;
        }
        MO_DERIVE(MorphologyFilter, Node);
            INPUT(SyncedMemory, input_image, nullptr);
            OUTPUT(SyncedMemory, output, SyncedMemory());
            PARAM(mo::EnumParameter, structuring_element_type, StructuringTypes());
            PARAM(mo::EnumParameter, morphology_type, MorphTypes());
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
