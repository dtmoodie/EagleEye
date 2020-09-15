#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/rcc/external_includes/cv_superres.hpp>

namespace aqdev
{
    class FrameSource : public cv::superres::FrameSource
    {
        cv::cuda::GpuMat m_current_frame;

      public:
        FrameSource();
        void nextFrame(cv::OutputArray frame) override;
        void reset() override;
        void inputFrame(cv::cuda::GpuMat);
    };

    class SuperResolution : public aq::nodes::Node
    {
        cv::Ptr<cv::superres::SuperResolution> super_res;
        cv::Ptr<FrameSource> frame_source;

      public:
        MO_DERIVE(SuperResolution, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            PARAM(int, scale, 2)
            PARAM(int, iterations, 50)
            PARAM(double, tau, 0.0)
            PARAM(double, lambda, 0.0)
            PARAM(double, alpha, 1.0)
            PARAM(int, kernel_size, 5)
            PARAM(int, blur_size, 5)
            PARAM(double, blur_sigma, 1.0)
            PARAM(int, temporal_radius, 1)
            OUTPUT(aq::SyncedImage, output)
        MO_END;

      protected:
        bool processImpl() override;
    };
} // namespace aqdev
