#pragma once
#include <Aquila/Nodes/Node.h>
#include "Aquila/rcc/external_includes/cv_cudafilters.hpp"
namespace aq
{
namespace Nodes
{
    class MedianBlur: public Node
    {
    public:
        MO_DERIVE(MedianBlur, Node)
            INPUT(SyncedMemory, input, nullptr)
            PARAM(int, window_size, 5)
            PARAM(int, partition, 128)
            OUTPUT(SyncedMemory, output, {})
        MO_END
    protected:
        bool ProcessImpl();

        cv::Ptr<cv::cuda::Filter> _median_filter;
    };
}
}
