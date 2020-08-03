#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "Channels.h"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <MetaObject/core/CvContext.hpp>
namespace aq
{
    namespace nodes
    {
        template <>
        bool ConvertToGrey::processImpl(mo::CvContext* ctx)
        {
            if (input->getSyncState() < input->DEVICE_UPDATED)
            {
                return processImpl(static_cast<mo::Context*>(ctx));
            }
            cv::cuda::GpuMat grey;
            cv::cuda::cvtColor(input->getGpuMat(stream()), grey, cv::COLOR_BGR2GRAY, 0, stream());
            grey_param.updateData(grey, mo::tag::_param = input_param);
            return true;
        }

        template <>
        bool Magnitude::processImpl(mo::CvContext* ctx)
        {
            if (input->getSyncState() < input->DEVICE_UPDATED)
            {
                return processImpl(static_cast<mo::Context*>(ctx));
            }
            cv::cuda::GpuMat magnitude;
            cv::cuda::magnitude(input->getGpuMat(stream()), magnitude, stream());
            output_param.updateData(magnitude, input_param.getTimestamp(), _ctx.get());
            return true;
        }

        template <>
        bool SplitChannels::processImpl(mo::CvContext* ctx)
        {
            std::vector<cv::cuda::GpuMat> _channels;
            cv::cuda::split(input->getGpuMat(stream()), _channels, stream());
            output_param.updateData(_channels, input_param.getTimestamp(), _ctx.get());
            return true;
        }

        template <>
        bool ConvertDataType::processImpl(mo::CvContext* ctx)
        {
            cv::cuda::GpuMat output;
            if (continuous)
            {
                cv::cuda::createContinuous(input->getSize(), data_type.current_selection, output);
            }
            input->getGpuMat(stream()).convertTo(output, data_type.current_selection, alpha, beta, stream());
            output_param.emitUpdate(input_param);
            return true;
        }

        template <>
        bool ConvertColorspace::processImpl(mo::CvContext* ctx)
        {
            cv::cuda::GpuMat output;
            cv::cuda::cvtColor(input_image->getGpuMat(stream()), output, conversion_code.getValue(), 0, stream());
            output_image_param.updateData(output, input_image_param.getTimestamp(), _ctx.get());
            return true;
        }

        template <>
        bool ConvertToHSV::processImpl(mo::CvContext* ctx)
        {
            cv::cuda::GpuMat output;
            cv::cuda::cvtColor(input_image->getGpuMat(ctx), output, cv::COLOR_BGR2HSV, 0, ctx->getStream());
            hsv_image_param.updateData(output, mo::tag::_param = input_image_param);
            return true;
        }

        template <>
        bool Reshape::processImpl(mo::CvContext* ctx)
        {
            reshaped_image_param.updateData(input_image->getGpuMat(ctx).reshape(channels, rows),
                                            mo::tag::_param = input_image_param);
            return true;
        }

    } // namespace nodes
} // namespace aq

#endif
