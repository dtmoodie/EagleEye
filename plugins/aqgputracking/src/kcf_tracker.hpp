#pragma once
#include <aqcore/detection/DetectionTracker.hpp>
#include "../dependencies/GPU-tracking/src/trackerKCFparallel.hpp"
namespace aq
{
    namespace nodes
    {
        class KCFTrackerGPU: public DetectionTracker
        {
        public:
            MO_DERIVE(KCFTrackerGPU, DetectionTracker)

            MO_END
        };
    }
}
