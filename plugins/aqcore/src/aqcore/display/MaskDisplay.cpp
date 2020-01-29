#include "MaskDisplay.hpp"
#include "Aquila/nodes/NodeInfo.hpp"

namespace aq
{
namespace nodes
{
    bool MaskOverlay::processImpl()
    {
        MO_ASSERT(image->getSize() == mask->getSize());
        if(image->getSyncState() < aq::SyncedMemory::DEVICE_UPDATED &&
                mask->getSyncState() << aq::SyncedMemory::DEVICE_UPDATED)
        {
            cv::cuda::GpuMat draw;
            image->clone(draw, stream());
            draw.setTo(color, mask->getGpuMat(stream()), stream());
            output_param.updateData(draw, mo::tag::_param = image_param);
        }else
        {
            // CPU
        }
        return true;
    }
}
}

using namespace aq::nodes;

MO_REGISTER_CLASS(MaskOverlay)
