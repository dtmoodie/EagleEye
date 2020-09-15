#include "super_resolution.h"

namespace aqdev
{

    FrameSource::FrameSource() {}

    void FrameSource::nextFrame(cv::OutputArray frame)
    {
        if (!m_current_frame.empty())
        {
            frame.getGpuMatRef() = m_current_frame;
        }
    }

    void FrameSource::reset() { m_current_frame = cv::cuda::GpuMat(); }

    void FrameSource::inputFrame(cv::cuda::GpuMat mat) { m_current_frame = std::move(mat); }

    bool SuperResolution::processImpl()
    {
        if (scale_param.getModified())
        {
            super_res->setScale(scale);
            scale_param.setModified(false);
        }
        if (iterations_param.getModified())
        {
            super_res->setIterations(iterations);
            iterations_param.setModified(false);
        }
        if (tau_param.getModified())
        {
            super_res->setTau(tau);
            tau_param.setModified(false);
        }
        if (lambda_param.getModified())
        {
            super_res->setLabmda(lambda);
            lambda_param.setModified(false);
        }
        if (alpha_param.getModified())
        {
            super_res->setAlpha(alpha);
            alpha_param.setModified(false);
        }
        if (kernel_size_param.getModified())
        {
            super_res->setKernelSize(kernel_size);
            kernel_size_param.setModified(false);
        }
        if (blur_size_param.getModified())
        {
            super_res->setBlurKernelSize(blur_size);
            blur_size_param.setModified(false);
        }
        if (blur_sigma_param.getModified())
        {
            super_res->setBlurSigma(blur_sigma);
            blur_sigma_param.setModified(false);
        }
        if (temporal_radius_param.getModified())
        {
            super_res->setTemporalAreaRadius(temporal_radius);
            temporal_radius_param.setModified(false);
        }
        cv::cuda::GpuMat result;

        // frame_source->input_frame(*input, stream());

        return true;
    }
} // namespace aqdev
// MO_REGISTER_CLASS(super_resolution)
