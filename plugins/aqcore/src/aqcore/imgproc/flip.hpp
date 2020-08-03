#pragma once
#include "Aquila/nodes/Node.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include <MetaObject/types/file_types.hpp>

namespace aq
{
    namespace nodes
    {
        class Flip : public Node
        {
          public:
            enum Axis
            {
                Diag = -1,
                X = 0,
                Y = 1
            };

            MO_DERIVE(Flip, Node)
                INPUT(SyncedImage, input)

                ENUM_PARAM(axis, X, Y, Diag)
                PARAM(cv::Rect2f, roi, {0.0F, 0.0F, 1.0F, 1.0F})

                OUTPUT(SyncedImage, output)
            MO_END;

            template <class CTX>
            bool processImpl(CTX* ctx);

          protected:
            bool processImpl();
        };
        class Rotate : public Node
        {
          public:
            MO_DERIVE(Rotate, Node)
                INPUT(SyncedImage, input)

                PARAM(int, angle_degrees, 180)

                OUTPUT(SyncedImage, output)
            MO_END
          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
