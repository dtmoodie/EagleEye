#pragma once
#include <aqcore_export.hpp>

#include "FeatureDetection.h"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class RegisterToReference : public Node
        {
            cv::Mat ref_keypoints;
            cv::cuda::GpuMat ref_descriptors;
            cv::cuda::GpuMat d_reference_grey;
            cv::cuda::GpuMat d_reference_original;
            cv::Ptr<cv::cuda::ORB> d_orb;
            cv::Ptr<cv::cuda::DescriptorMatcher> d_matcher;

          public:
            MO_DERIVE(RegisterToReference, Node)
                INPUT(SyncedImage, reference_image)
                INPUT(SyncedImage, current_image)
            MO_END

          protected:
            bool processImpl();
        };

    } // namespace nodes
} // namespace aq
