#pragma once
#include "../precompiled.hpp"
#include <Aquila/types/DetectionDescription.hpp>
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
                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, output)
            MO_END;

          protected:
            bool processImpl() override;
        };

        class AutoScale : public Node
        {
          public:
            MO_DERIVE(AutoScale, Node)
                INPUT(SyncedImage, input_image)
                OUTPUT(SyncedImage, output_image)
            MO_END;

          protected:
            bool processImpl() override;
        };

        class Normalize : public Node
        {
          public:
            MO_DERIVE(Normalize, Node)
                INPUT(SyncedImage, input_image)
                OPTIONAL_INPUT(SyncedImage, mask)

                ENUM_PARAM(norm_type, cv::NORM_MINMAX, cv::NORM_L2, cv::NORM_L1, cv::NORM_INF)
                PARAM(double, alpha, 0)
                PARAM(double, beta, 1)

                OUTPUT(SyncedImage, normalized_output)
            MO_END;

          protected:
            bool processImpl() override;
        };
    } // namespace nodes
} // namespace aq
