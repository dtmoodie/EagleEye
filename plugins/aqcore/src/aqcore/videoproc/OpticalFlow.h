#pragma once
#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeContextSwitch.hpp>

#include <Aquila/rcc/external_includes/cv_cudaoptflow.hpp>
#include <Aquila/rcc/external_includes/cv_video.hpp>

#include <MetaObject/params/TMultiSubscriber.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class IPyrOpticalFlow : public Node
        {
          public:
            IPyrOpticalFlow();
            MO_DERIVE(IPyrOpticalFlow, Node)
                INPUT(SyncedImage, input)
                OPTIONAL_INPUT(std::vector<cv::cuda::GpuMat>, image_pyramid)

                PARAM(int, window_size, 13)
                PARAM(int, iterations, 30)
                PARAM(int, pyramid_levels, 3)
                PARAM(bool, use_initial_flow, false)
            MO_END;

          protected:
            size_t prepPyramid();
            void build_pyramid(std::vector<cv::cuda::GpuMat>& pyramid);

            mo::OptionalTime m_prev_time;
            std::vector<cv::cuda::GpuMat> m_prev_grey_pyramid;
            std::vector<cv::cuda::GpuMat> m_grey_pyramid;
        };

        class DensePyrLKOpticalFlow : public IPyrOpticalFlow
        {
          public:
            MO_DERIVE(DensePyrLKOpticalFlow, IPyrOpticalFlow)
                OUTPUT(SyncedImage, flow_field)
            MO_END;
            bool processImpl();

          protected:
            cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> m_opt_flow;
        };

        class SparsePyrLKOpticalFlow : public IPyrOpticalFlow
        {
          public:
            MO_DERIVE(SparsePyrLKOpticalFlow, IPyrOpticalFlow)
                FLAGGED_INPUT(mo::ParamFlags::kREQUIRE_BUFFERED, SyncedImage, input_points)

                OUTPUT(SyncedImage, tracked_points)
                OUTPUT(SyncedImage, status)
                OUTPUT(SyncedImage, error)
            MO_END;

          protected:
            bool processImpl();
            cv::cuda::GpuMat m_prev_key_points;
            mo::OptionalTime m_prev_time;
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> m_opt_flow;
        };

        class PyrLKLandmarkTracker : public Node
        {
          public:
            using Components_t = ct::VariadicTypedef<aq::detection::LandmarkDetection>;
            using LandmarkDetectionSet_t = aq::TEntityComponentSystem<Components_t>;

            MO_DERIVE(PyrLKLandmarkTracker, Node)
                INPUT(SyncedImage, input)
                FLAGGED_INPUT(mo::ParamFlags::kREQUIRE_BUFFERED, LandmarkDetectionSet_t, detections)

                PARAM(int, window_size, 13)
                PARAM(int, iterations, 30)
                PARAM(int, pyramid_levels, 3)

                OUTPUT(LandmarkDetectionSet_t, output)
            MO_END;

            template <class STREAM>
            bool processImpl(STREAM& stream);

          protected:
            bool processImpl() override;

            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_opt_flow;
            std::vector<aq::SyncedImage> m_prev_pyramid;
            mo::OptionalTime m_prev_time;
        };

    } // namespace nodes
} // namespace aq
