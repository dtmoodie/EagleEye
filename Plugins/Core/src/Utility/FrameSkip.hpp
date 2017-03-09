#pragma once

#include <Aquila/Nodes/Node.h>

namespace aq
{
    namespace Nodes
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
            bool ProcessImpl();
            int frame_count = 0;
        };
    }
}
