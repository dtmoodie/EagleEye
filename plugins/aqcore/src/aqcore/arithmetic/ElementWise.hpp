#pragma once

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
    namespace nodes
    {
        class Equal : public Node
        {
          public:
            MO_DERIVE(Equal, Node)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(double, value, 0)
                OUTPUT(SyncedMemory, output, {})
            MO_END
          protected:
            bool processImpl();
        };
        class AddBinary : public Node
        {
          public:
            MO_DERIVE(AddBinary, Node)
                INPUT(SyncedMemory, in1, nullptr)
                INPUT(SyncedMemory, in2, nullptr)
                PARAM(double, weight1, 1.0)
                PARAM(double, weight2, 1.0)
                OUTPUT(SyncedMemory, output, {})
            MO_END
          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
