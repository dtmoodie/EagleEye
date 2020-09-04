#pragma once
#include "Aquila/rcc/external_includes/cv_cudafeatures2d.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class MinMax : public Node
        {
          public:
            MO_DERIVE(MinMax, Node)
                INPUT(SyncedImage, input)

                OUTPUT(double, min_value, 0.0)
                OUTPUT(double, max_value, 0.0)
            MO_END

          protected:
            bool processImpl();
        };

        class Threshold : public Node
        {
          public:
            MO_DERIVE(Threshold, Node)
                INPUT(SyncedImage, input)
                OPTIONAL_INPUT(double, input_max)
                OPTIONAL_INPUT(double, input_min)

                PARAM(double, replace_value, 255.0)
                PARAM(double, max, 1.0)
                PARAM(double, min, 0.0)
                PARAM(bool, two_sided, false)
                PARAM(bool, truncate, false)
                PARAM(bool, inverse, false)
                PARAM(bool, source_value, true)
                PARAM(double, input_percent, 0.9)

                OUTPUT(SyncedImage, mask)

            MO_END

          protected:
            bool processImpl();
        };

        class NonMaxSuppression : public Node
        {
          public:
            MO_DERIVE(NonMaxSuppression, Node)
                PARAM(int, size, 5)
                INPUT(SyncedImage, input)
                INPUT(SyncedImage, mask)
                OUTPUT(SyncedImage, suppressed_output)
            MO_END
        };

    } // namespace nodes
} // namespace aq