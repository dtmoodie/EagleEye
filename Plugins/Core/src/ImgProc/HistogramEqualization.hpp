#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace aq
{
    namespace nodes
    {
        class HistogramEqualization: public Node
        {
        public:
            MO_DERIVE(HistogramEqualization, Node)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(bool, per_channel, false)
                OUTPUT(SyncedMemory, output, {})
            MO_END
        protected:
            bool processImpl();
        };
        class CLAHE: public Node
        {
        public:
            MO_DERIVE(CLAHE, Node)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(double, clip_limit, 40)
                PARAM(int, grid_size, 8)
                OUTPUT(SyncedMemory, output, {})
            MO_END
        protected:
            bool processImpl();
            cv::Ptr<cv::cuda::CLAHE> _clahe;
        };
    }
}
