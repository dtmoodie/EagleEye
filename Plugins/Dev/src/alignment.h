#pragma once
#include "Aquila/nodes/Node.hpp"
#include <Aquila/types/Stamped.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <boost/circular_buffer.hpp>

namespace aq{
namespace Nodes{
        class TrackCameraMotion : public Node{
        public:
        protected:
            virtual bool processImpl();
            boost::circular_buffer<TS<cv::Mat>> relative_motions;
        };

} // namespace aq::Nodes
} // namespace a
