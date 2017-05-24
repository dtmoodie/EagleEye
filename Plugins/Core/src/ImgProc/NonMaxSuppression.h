#pragma once

#include "src/precompiled.hpp"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace Nodes
    {
        class MinMax: public Node
        {
        public:
            MO_DERIVE(MinMax, Node)
                INPUT(SyncedMemory, input, nullptr);
                OUTPUT(double, min_value, 0.0);
                OUTPUT(double, max_value, 0.0);
            MO_END;
        protected:
            bool processImpl();
            
        };
        class Threshold : public Node
        {
        public:
            MO_DERIVE(Threshold, Node)
                OPTIONAL_INPUT(double, input_max, nullptr);
                OPTIONAL_INPUT(double, input_min, nullptr);
                PARAM(double, replace_value, 255.0);
                PARAM(double, max, 1.0);
                PARAM(double, min, 0.0);
                PARAM(bool, two_sided, false);
                PARAM(bool, truncate, false);
                PARAM(bool, inverse, false);
                PARAM(bool, source_value, true);
                OUTPUT(SyncedMemory, mask, SyncedMemory());
                PARAM(double, input_percent, 0.9);
                INPUT(SyncedMemory, input, nullptr);
            MO_END;
        protected:
            bool processImpl();
        };

        class NonMaxSuppression: public Node
        {
        public:
            MO_DERIVE(NonMaxSuppression, Node)
                PARAM(int, size, 5);
                INPUT(SyncedMemory, input, nullptr);
                INPUT(SyncedMemory, mask, nullptr);
                OUTPUT(SyncedMemory, suppressed_output, SyncedMemory());
            MO_END;
        };
    }
}
