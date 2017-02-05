#pragma once
#include "ROSExport.hpp"
#include <EagleLib/Nodes/IFrameGrabber.hpp>

namespace EagleLib
{
    namespace Nodes
    {
        class ROS_EXPORT RosSubscriber : public FrameGrabberThreaded
        {
        public:
            MO_DERIVE(RosSubscriber, FrameGrabberThreaded)


            MO_END

        };
    }
}
