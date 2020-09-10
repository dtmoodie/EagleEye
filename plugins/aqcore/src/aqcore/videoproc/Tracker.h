#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/params/TSubscriber.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class KeyFrameTracker : public Node
        {
          public:
            MO_DERIVE(KeyFrameTracker, Node)
                INPUT(SyncedImage, input_image)
                INPUT(SyncedImage, input_mask)

                PARAM(int, frames_to_track, 5)
                PARAM(double, upper_quality, 0.7)
                PARAM(double, lower_quality, 0.4)
                PARAM(int, min_keypoints, 200)
            MO_END;

          protected:
            bool processImpl();
        };

        class CMT : public Node
        {
          public:
            MO_DERIVE(CMT, Node)
            MO_END;

          protected:
            bool processImpl();
        };

        class TLD : public Node
        {
          public:
            MO_DERIVE(TLD, Node)
            MO_END;

          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
