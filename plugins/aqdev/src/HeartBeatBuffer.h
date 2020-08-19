#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include "Aquila/nodes/Node.hpp"

#include <boost/circular_buffer.hpp>

namespace aq
{
    namespace nodes
    {
        class HeartBeatBuffer : public Node
        {
            boost::circular_buffer<SyncedImage> image_buffer;
            time_t lastTime;
            bool activated;

          public:
        };
    } // namespace nodes
} // namespace aq
