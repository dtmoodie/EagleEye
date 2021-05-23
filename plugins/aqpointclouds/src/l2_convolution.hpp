#pragma once
#include "aqpointclouds_export.hpp"

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <aqcore/OpenCVCudaNode.hpp>
#include <opencv2/core/cuda.hpp>

#include <Aquila/rcc/external_includes/cv_core.hpp>

namespace aq
{
    namespace pointclouds
    {
        class ConvolutionL2 : public aqcore::OpenCVCudaNode
        {
          public:
            MO_DERIVE(ConvolutionL2, aqcore::OpenCVCudaNode)
                INPUT(aq::SyncedImage, input)
                PARAM(int, kernel_size, 3)
                PARAM(int, distance_threshold, 1.0f)
                OUTPUT(aq::SyncedImage, distance, {})
                OUTPUT(aq::SyncedImage, index, {})
            MO_END;

          protected:
            bool processImpl() override;
        };

        class ConvolutionL2ForegroundEstimate : public aqcore::OpenCVCudaNode
        {
          public:
            MO_DERIVE(ConvolutionL2ForegroundEstimate, aqcore::OpenCVCudaNode)
                INPUT(aq::SyncedImage, input)
                PARAM(int, kernel_size, 3)
                PARAM(int, distance_threshold, 1.0f)
                PARAM(bool, build_model, false)
                MO_SLOT(void, buildModel)
                OUTPUT(aq::SyncedImage, distance, {})
                OUTPUT(aq::SyncedImage, index, {})
            MO_END;

          protected:
            bool processImpl() override;
            cv::cuda::GpuMat prev;
        };
    } // namespace pointclouds
} // namespace aq
