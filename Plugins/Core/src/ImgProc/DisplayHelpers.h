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
            INPUT(SyncedMemory, image, nullptr)
            INPUT(std::vector<std::string>, labels, nullptr)
            OPTIONAL_INPUT(std::vector<DetectedObject>, detections, nullptr)
            PROPERTY(std::vector<cv::Vec3b>, colors, std::vector<cv::Vec3b>())
            OUTPUT(SyncedMemory, image_with_detections, SyncedMemory())
        MO_END
    protected:
        bool ProcessImpl();
    };
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
