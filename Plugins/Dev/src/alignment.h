#pragma once
#include "Aquila/Nodes/Node.h"
#include <Aquila/types/Stamped.hpp>
#include <boost/circular_buffer.hpp>

namespace aq
{
    namespace Nodes
    {
        class TrackCameraMotion : public Node
        {
            boost::circular_buffer<TS<cv::Mat>> relative_motions;

        public:
            virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);

        };

    }

}
