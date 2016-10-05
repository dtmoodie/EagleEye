#pragma once
#include <src/precompiled.hpp>
#include "FeatureDetection.h"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib
{ 
    namespace Nodes
    {
        class RegisterToReference: public Node
        {
            cv::Mat ref_keypoints;
            cv::cuda::GpuMat ref_descriptors;
            cv::cuda::GpuMat d_reference_grey;
            cv::cuda::GpuMat d_reference_original;
            cv::Ptr<cv::cuda::ORB> d_orb;
            cv::Ptr<cv::cuda::DescriptorMatcher> d_matcher;
        public:
            MO_DERIVE(RegisterToReference, Node)
                INPUT(SyncedMemory, reference_image, nullptr);
                INPUT(SyncedMemory, current_image, nullptr);
                
            MO_END;
        protected:
            bool ProcessImpl();
        };
    } // namespace nodes
} // namespace EagleLib 
