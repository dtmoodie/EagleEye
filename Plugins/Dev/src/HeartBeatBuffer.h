#pragma once
#include "Aquila/nodes/Node.hpp"
#include <Aquila/types/Stamped.hpp>
#include <boost/circular_buffer.hpp>

namespace aq
{
    namespace Nodes
    {
    class HeartBeatBuffer : public Node
    {
        boost::circular_buffer<TS<SyncedMemory>> image_buffer;
        time_t lastTime;
        bool activated;
    public:
    };
    }
}
