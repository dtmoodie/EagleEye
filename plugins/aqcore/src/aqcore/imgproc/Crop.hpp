#pragma once
#include <aqcore_export.hpp>

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

namespace aq
{
    namespace nodes
    {
        class Crop : public Node
        {
          public:
            MO_DERIVE(Crop, Node)
                INPUT(SyncedImage, input)
                PARAM(cv::Rect2f, roi, {0.0F, 0.0F, 1.0F, 1.0F})
                OUTPUT(SyncedImage, output)
            MO_END;

          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
