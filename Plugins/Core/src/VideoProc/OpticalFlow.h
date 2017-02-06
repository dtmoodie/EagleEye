#pragma once
#include <src/precompiled.hpp>

#include <EagleLib/rcc/external_includes/cv_cudaoptflow.hpp>
#include "EagleLib/utilities/CudaUtils.hpp"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
        class IPyrOpticalFlow: public Node
        {
        public:
            IPyrOpticalFlow();
            MO_DERIVE(IPyrOpticalFlow, Node)
                INPUT(SyncedMemory, input, nullptr)
                OPTIONAL_INPUT(std::vector<cv::cuda::GpuMat>, image_pyramid, nullptr)
                PARAM(int, window_size, 13)
                PARAM(int, iterations, 30)
                PARAM(int, pyramid_levels, 3)
                PARAM(bool, use_initial_flow, false)
            MO_END;
        protected:
            long long PrepPyramid();
            void build_pyramid(std::vector<cv::cuda::GpuMat>& pyramid);
            TS<std::vector<cv::cuda::GpuMat>> prevGreyImg;
            std::vector<cv::cuda::GpuMat> greyImg;
        };
        class DensePyrLKOpticalFlow : public IPyrOpticalFlow
        {
        public:
            MO_DERIVE(DensePyrLKOpticalFlow, IPyrOpticalFlow)
                OUTPUT(SyncedMemory, flow_field, SyncedMemory());
            MO_END;
            bool ProcessImpl();
        protected:
            cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> opt_flow;
        };

        class SparsePyrLKOpticalFlow : public IPyrOpticalFlow
        {
        public:
            MO_DERIVE(SparsePyrLKOpticalFlow, IPyrOpticalFlow)
                INPUT(SyncedMemory, input_points, nullptr);
                APPEND_FLAGS(input_points, mo::Buffer_e);
                OUTPUT(SyncedMemory, tracked_points, SyncedMemory());
                OUTPUT(SyncedMemory, status, SyncedMemory());
                OUTPUT(SyncedMemory, error, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
            cv::cuda::GpuMat prev_key_points;
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optFlow;
        };
    }
}
