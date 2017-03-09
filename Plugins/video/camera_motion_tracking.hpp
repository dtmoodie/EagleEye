#pragma once

#include "Aquila/Nodes/Node.h"
#include <MetaObject/MetaObject.hpp>
namespace aq
{
    namespace Nodes
    {
        class track_camera_motion : public Node
        {
        public:
            MO_DERIVE(track_camera_motion, Node)

            MO_END;
            static std::vector<std::vector<std::string>> GetParentalDependencies();
            bool ProcessImpl();
        };
    }
}
