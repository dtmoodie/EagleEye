#include "FrameSkip.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

using namespace aq::Nodes;

bool FrameSkip::processImpl()
{
    ++frame_count;
    if(frame_count > frame_skip)
    {
        output_param.updateData(*input, input_param.getTimestamp(), _ctx);
        frame_count = 0;
        return true;
    }
    return true;
}

MO_REGISTER_CLASS(FrameSkip)
