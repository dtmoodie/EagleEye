#pragma once
#include "Aquila/nodes/Node.hpp"
#include <Aquila/types/Stamped.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <boost/circular_buffer.hpp>

namespace aq
{
    namespace nodes
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
