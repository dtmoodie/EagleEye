#pragma once
#include "EagleLib/nodes/Node.h"
#include <boost/circular_buffer.hpp>

namespace EagleLib
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