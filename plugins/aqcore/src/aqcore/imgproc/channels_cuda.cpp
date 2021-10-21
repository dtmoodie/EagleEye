#include <MetaObject/core/metaobject_config.hpp>
#include <opencv2/cvconfig.h>
#ifdef HAVE_CUDA
#include "Channels.h"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>

#include <Aquila/types/CVStream.hpp>

namespace aqcore
{

    template <>
    bool ConvertToGrey::processImpl(mo::IDeviceStream& stream)
    {
        cv::cuda::GpuMat input_mat = input->getGpuMat(&stream);
        cv::cuda::GpuMat grey;
        aq::CVStream& cv_stream = dynamic_cast<aq::CVStream&>(stream);
        cv::cuda::cvtColor(input_mat, grey, cv::COLOR_BGR2GRAY, 0, cv_stream.getCVStream());
        this->output.publish(aq::SyncedImage(grey), mo::tags::param = &input_param);
        return true;
    }

    template <>
    bool Magnitude::processImpl(mo::IDeviceStream& stream)
    {
        cv::cuda::GpuMat input_mat = input->getGpuMat(&stream);
        cv::cuda::GpuMat magnitude;
        aq::CVStream& cv_stream = dynamic_cast<aq::CVStream&>(stream);
        cv::cuda::magnitude(input_mat, magnitude, cv_stream.getCVStream());
        this->output.publish(aq::SyncedImage(magnitude), mo::tags::param = &input_param);
        return true;
    }

    /*template <>
    bool SplitChannels::processImpl(mo::IDeviceStream& stream)
    {
        std::vector<cv::cuda::GpuMat> _channels;
        cv::cuda::split(input->getGpuMat(stream()), _channels, stream());
        output_param.updateData(_channels, input_param.getTimestamp(), _ctx.get());
        return true;
    }*/

    template <>
    bool ConvertDataType::processImpl(mo::IDeviceStream& stream)
    {
        cv::cuda::GpuMat output;
        cv::cuda::GpuMat input_mat = input->getGpuMat(&stream);
        if (continuous)
        {
            cv::cuda::createContinuous(input_mat.size(), data_type.current_selection, output);
        }
        aq::CVStream& cv_stream = dynamic_cast<aq::CVStream&>(stream);

        input_mat.convertTo(output, data_type.current_selection, alpha, beta, cv_stream.getCVStream());
        this->output.publish(aq::SyncedImage(output), mo::tags::param = &input_param);
        return true;
    }

    template <>
    bool ConvertColorspace::processImpl(mo::IDeviceStream& stream)
    {
        cv::cuda::GpuMat output;
        cv::cuda::GpuMat input_mat = input->getGpuMat(&stream);
        aq::CVStream& cv_stream = dynamic_cast<aq::CVStream&>(stream);
        cv::cuda::cvtColor(input_mat, output, conversion_code.getValue(), 0, cv_stream.getCVStream());
        this->output.publish(aq::SyncedImage(output), mo::tags::param = &input_param);
        return true;
    }

    template <>
    bool ConvertToHSV::processImpl(mo::IDeviceStream& stream)
    {
        cv::cuda::GpuMat output;
        aq::CVStream& cv_stream = dynamic_cast<aq::CVStream&>(stream);
        cv::cuda::GpuMat input = this->input->getGpuMat(&stream);
        cv::cuda::cvtColor(input, output, cv::COLOR_BGR2HSV, 0, cv_stream.getCVStream());
        this->output.publish(aq::SyncedImage(output), mo::tags::param = &input_param);
        return true;
    }

    template <>
    bool Reshape::processImpl(mo::IDeviceStream& stream)
    {
        cv::cuda::GpuMat output;
        cv::cuda::GpuMat input = this->input->getGpuMat(&stream);
        output = input.reshape(channels, rows);
        this->output.publish(aq::SyncedImage(output), mo::tags::param = &input_param);
        return true;
    }

} // namespace aqcore

#endif
