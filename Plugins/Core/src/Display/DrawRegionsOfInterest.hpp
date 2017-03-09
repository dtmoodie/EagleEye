#pragma once

#include "Aquila/Nodes/Node.h"

#include "Aquila/utilities/GpuDrawing.hpp"

namespace aq
{
namespace Nodes
{
    class DrawRegionsOfInterest: public Node
    {
    public:
        MO_DERIVE(DrawRegionsOfInterest, Node)
            INPUT(SyncedMemory, image, nullptr)
            INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
            OUTPUT(SyncedMemory, output, {})
        MO_END;
    protected:
        bool ProcessImpl();
    };
}
}
