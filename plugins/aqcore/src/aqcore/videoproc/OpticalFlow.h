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
            MO_DERIVE(IPyrOpticalFlow, Node)
                INPUT(SyncedImage, image)

                PARAM(int, window_size, 13)
                PARAM(int, iterations, 30)
                PARAM(int, pyramid_levels, 3)
                PARAM(bool, use_initial_flow, false)
            MO_END;

            std::vector<cv::cuda::GpuMat> makePyramid(const cv::cuda::GpuMat& mat, cv::cuda::Stream&) const;
            std::vector<cv::Mat> makePyramid(const cv::Mat& mat) const;

          private:
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
                OUTPUT(SyncedImage, tracked_points)
                OUTPUT(SyncedImage, status)
                OUTPUT(SyncedImage, error)
            MO_END;

          protected:
        };

        class DetectionLandmarkTracker : public SparsePyrLKOpticalFlow
        {
          public:
            using Components_t = ct::VariadicTypedef<aq::detection::LandmarkDetection>;
            using LandmarkDetectionSet_t = aq::TDetectedObjectSet<Components_t>;

            MO_DERIVE(DetectionLandmarkTracker, SparsePyrLKOpticalFlow)
                INPUT(SyncedImage, image)
                FLAGGED_INPUT(mo::ParamFlags::kREQUIRE_BUFFERED, LandmarkDetectionSet_t, detections)

                OUTPUT(LandmarkDetectionSet_t, output)
            MO_END;

            bool processImpl(mo::IAsyncStream& stream) override;
            bool processImpl(mo::IDeviceStream& stream) override;
            bool processImpl() override;

          private:
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_opt_flow;
            std::vector<cv::Mat> m_prev_cpu_pyramid;
            std::vector<cv::cuda::GpuMat> m_prev_gpu_pyramid;
            boost::optional<mo::Header> m_previous_header;
        };

    } // namespace nodes
} // namespace aq
