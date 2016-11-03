#pragma once

#include "EagleLib/Nodes/Node.h"
#include <MetaObject/MetaObject.hpp>
namespace EagleLib
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
