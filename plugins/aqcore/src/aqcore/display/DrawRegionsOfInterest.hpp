#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/utilities/GpuDrawing.hpp>

namespace aq
{
    namespace nodes
    {
        class DrawRegionsOfInterest : public Node
        {
          public:
            MO_DERIVE(DrawRegionsOfInterest, Node)
                INPUT(SyncedMemory, image, nullptr)
                INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
                OUTPUT(SyncedMemory, output, {})
            MO_END;

          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
