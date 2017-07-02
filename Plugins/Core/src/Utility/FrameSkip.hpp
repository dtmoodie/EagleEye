#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
    namespace nodes
    {
        class FrameSkip: public Node
        {
        public:
            MO_DERIVE(FrameSkip, Node)
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(SyncedMemory, output, {})
                PARAM(int, frame_skip, 30)
            MO_END
        protected:
            bool processImpl();
            int frame_count = 0;
        };
    }
}
