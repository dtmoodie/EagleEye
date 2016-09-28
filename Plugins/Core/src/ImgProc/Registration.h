#pragma once
#include "EagleLib/nodes/Node.h"
#include <opencv2/cudafeatures2d.hpp>
#include <EagleLib/rcc/external_includes/cv_cudafeatures2d.hpp>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib{ namespace Nodes{
    class register_to_reference: public Node
    {
        cv::Mat ref_keypoints;
        cv::cuda::GpuMat ref_descriptors;
        cv::cuda::GpuMat d_reference_grey;
        cv::cuda::GpuMat d_reference_original;
        cv::Ptr<cv::cuda::ORB> d_orb;
        cv::Ptr<cv::cuda::DescriptorMatcher> d_matcher;
    public:
        register_to_reference();
        virtual void doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
        virtual void NodeInit(bool firstInit);

    };



} /* namespace nodes */} /* namespace EagleLib */
