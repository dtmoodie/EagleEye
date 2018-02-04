#pragma once
#include "../precompiled.hpp"
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/utilities/ColorMapping.hpp>

namespace aq
{
    namespace nodes
    {
        class Scale : public Node
        {
          public:
            MO_DERIVE(Scale, Node)
                PARAM(double, scale_factor, 1.0)
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(SyncedMemory, output, SyncedMemory())
            MO_END
          protected:
            bool processImpl();
        };

        class AutoScale : public Node
        {
          public:
            MO_DERIVE(AutoScale, Node)
                INPUT(SyncedMemory, input_image, nullptr)
                OUTPUT(SyncedMemory, output_image, SyncedMemory())
            MO_END
          protected:
            bool processImpl();
        };

        class IDrawDetections : public Node
        {
          public:
            typedef std::map<std::string, cv::Vec3b> Colormap_t;
            MO_DERIVE(IDrawDetections, Node)
                INPUT(std::vector<std::string>, labels, nullptr)
                // APPEND_FLAGS(labels, mo::Desynced_e)
                PROPERTY(std::vector<cv::Vec3b>, colors, std::vector<cv::Vec3b>())
                PARAM(Colormap_t, colormap, {})
            MO_END;

          protected:
            void createColormap();
        };

        class DrawDetections : public IDrawDetections
        {
          public:
            MO_DERIVE(DrawDetections, IDrawDetections)
                INPUT(SyncedMemory, image, nullptr)
                INPUT(DetectedObjectSet, detections, nullptr)
                PARAM(bool, draw_class_label, true)
                PARAM(bool, draw_detection_id, true)
                OUTPUT(SyncedMemory, output, SyncedMemory())
            MO_END
          protected:
            bool processImpl();
        };

        class Normalize : public Node
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
            bool processImpl();
        };
    }
}
