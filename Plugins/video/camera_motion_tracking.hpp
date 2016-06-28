#pragma once

#include "EagleLib/nodes/Node.h"

namespace EagleLib
{
    namespace Nodes
    {
        class track_camera_motion : public Node
        {
        public:
            class track_camera_motion_info : public NodeInfo
            {
            public:
                track_camera_motion_info();
                virtual std::vector<std::vector<std::string>> GetParentalDependencies() const;
            };

            TS<SyncedMemory> doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream);

        };
    }
}