#pragma once

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
    namespace pointclouds
    {
        class ConvolutionL2 : public nodes::Node
        {
          public:
            MO_DERIVE(ConvolutionL2, nodes::Node)
                INPUT(aq::SyncedMemory, input)
                PARAM(int, kernel_size, 3)
                PARAM(int, distance_threshold, 1.0f)
                OUTPUT(aq::SyncedMemory, distance, {})
                OUTPUT(aq::SyncedMemory, index, {})
            MO_END

          protected:
            bool processImpl();
        };

        class ConvolutionL2ForegroundEstimate : public nodes::Node
        {
          public:
            MO_DERIVE(ConvolutionL2ForegroundEstimate, nodes::Node)
                INPUT(aq::SyncedMemory, input)
                PARAM(int, kernel_size, 3)
                PARAM(int, distance_threshold, 1.0f)
                PARAM(bool, build_model, false)
                MO_SLOT(void, buildModel)
                OUTPUT(aq::SyncedMemory, distance, {})
                OUTPUT(aq::SyncedMemory, index, {})
            MO_END

          protected:
            bool processImpl();
            cv::cuda::GpuMat prev;
        };
    } // namespace pointclouds
} // namespace aq
