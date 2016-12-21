#pragma once
#include <src/precompiled.hpp>
#include <EagleLib/ObjectDetection.hpp>

#include <EagleLib/utilities/ColorMapping.hpp>
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib
{
    namespace Nodes
    {
    class Scale:public Node
    {
    public:
        MO_DERIVE(Scale, Node)
            PARAM(double, scale_factor, 1.0)
            INPUT(SyncedMemory, input, nullptr)
            OUTPUT(SyncedMemory, output, SyncedMemory())
        MO_END
    protected:
        bool ProcessImpl();
    };

    class AutoScale: public Node
    {
    public:
    MO_DERIVE(AutoScale, Node)
        INPUT(SyncedMemory, input_image, nullptr)
        OUTPUT(SyncedMemory, output_image, SyncedMemory())
    MO_END
    protected:
        bool ProcessImpl();
    };
    class DrawDetections: public Node
    {
    public:
        MO_DERIVE(DrawDetections, Node)
            PROPERTY(std::vector<std::string>, labels, std::vector<std::string>())
            PROPERTY(std::vector<cv::Vec3b>, colors, std::vector<cv::Vec3b>())
            INPUT(SyncedMemory, image, nullptr)
            OPTIONAL_INPUT(std::vector<DetectedObject>, detections, nullptr)
            PARAM(mo::ReadFile, detection_list, mo::ReadFile(""))
            OUTPUT(SyncedMemory, image_with_detections, SyncedMemory())
        MO_END
    protected:
        bool ProcessImpl();
    };
    /*class Colormap: public Node
    {
    protected:
        cv::cuda::GpuMat color_mapped_image;
        color_mapper mapper;
    public:
        void Rescale();
        bool rescale;
        Colormap();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };*/
    /*class QtColormapDisplay: public Colormap
    {
    public:
        void display();
        QtColormapDisplay();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };*/
    class Normalize: public Node
    {
    public:
        MO_DERIVE(Normalize, Node)
            INPUT(SyncedMemory, input_image, nullptr);
            OPTIONAL_INPUT(SyncedMemory, mask, nullptr);
            OUTPUT(SyncedMemory, normalized_output, SyncedMemory());
            ENUM_PARAM(norm_type, cv::NORM_MINMAX, cv::NORM_L2, cv::NORM_L1, cv::NORM_INF);
            PARAM(double, alpha, 0);
            PARAM(double, beta, 1);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    }
}
