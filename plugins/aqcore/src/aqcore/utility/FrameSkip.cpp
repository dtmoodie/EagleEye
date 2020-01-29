#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "FrameSkip.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

using namespace aq::nodes;

bool FrameSkip::processImpl()
{
    ++frame_count;
    if (frame_count > frame_skip)
    {
        output_param.updateData(*input, input_param.getTimestamp(), _ctx.get());
        frame_count = 0;
        return true;
    }
    return true;
}

MO_REGISTER_CLASS(FrameSkip)
#endif