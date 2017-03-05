#pragma once

#include "EagleLib/Nodes/Node.h"

#include "EagleLib/utilities/GpuDrawing.hpp"

namespace EagleLib
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
