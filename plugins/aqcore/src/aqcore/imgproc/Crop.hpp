#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
namespace nodes
{
class Crop : public Node
{
  public:
    MO_DERIVE(Crop, Node)
        INPUT(SyncedMemory, input, nullptr)
        PARAM(cv::Rect2f, roi, {0.0F, 0.0F, 1.0F, 1.0F})
        OUTPUT(SyncedMemory, output, {})
    MO_END
  protected:
    bool processImpl();
};
}
}
