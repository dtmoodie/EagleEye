#pragma once
#include "ROSExport.hpp"
#include <Aquila/Nodes/IFrameGrabber.hpp>

namespace aq
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
