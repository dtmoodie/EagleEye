#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
    namespace nodes
    {
        class MaskOverlay: public Node
        {
        public:
            MO_DERIVE(MaskOverlay, Node)
                INPUT(aq::SyncedMemory, image, nullptr)
                INPUT(aq::SyncedMemory, mask, nullptr)
                PARAM(cv::Scalar, color, {255,0,0})
                OUTPUT(aq::SyncedMemory, output, {})
            MO_END
            protected:
                virtual bool processImpl() override;
        };
    }
}
