#pragma once

#include "Aquila/nodes/Node.hpp"
#include <MetaObject/object/MetaObject.hpp>
namespace aq
{
    namespace nodes
    {
        class track_camera_motion : public Node
        {
        public:
            MO_DERIVE(track_camera_motion, Node)

            MO_END;
            static std::vector<std::vector<std::string>> GetParentalDependencies();
            bool processImpl();
        };
    }
}
