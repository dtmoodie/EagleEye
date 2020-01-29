#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "DisplayHelpers.h"

#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <Aquila/rcc/external_includes/aqcore_link_libs.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <fstream>

using namespace aq;
using namespace aq::nodes;

bool Scale::processImpl()
{
    cv::cuda::GpuMat scaled;
    cv::cuda::multiply(input->getGpuMat(stream()), cv::Scalar(scale_factor), scaled, 1, -1, stream());
    output_param.updateData(scaled, input_param.getTimestamp(), _ctx.get());
    return true;
}


bool AutoScale::processImpl()
{
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(input_image->getGpuMat(stream()), channels, stream());
    for (size_t i = 0; i < channels.size(); ++i)
    {
        double minVal, maxVal;
        cv::cuda::minMax(channels[i], &minVal, &maxVal);
        double scaleFactor = 255.0 / (maxVal - minVal);
        channels[i].convertTo(channels[0], CV_8U, scaleFactor, minVal * scaleFactor);
    }
    cv::cuda::merge(channels, output_image.getGpuMat(stream()), stream());
    return true;
}

bool Normalize::processImpl()
{
    cv::cuda::GpuMat normalized;

    if (input_image->getChannels() == 1)
    {
        cv::cuda::normalize(input_image->getGpuMat(stream()),
                            normalized,
                            alpha,
                            beta,
                            static_cast<int>(norm_type.current_selection),
                            input_image->getDepth(),
                            mask == NULL ? cv::noArray() : mask->getGpuMat(stream()),
                            stream());
        normalized_output_param.updateData(normalized, input_image_param.getTimestamp(), _ctx.get());
        return true;
    }
    else
    {
        std::vector<cv::cuda::GpuMat> channels;

        if (input_image->getNumMats() == 1)
        {
            cv::cuda::split(input_image->getGpuMat(stream()), channels, stream());
        }
        else
        {
            channels = input_image->getGpuMatVec(stream());
        }
        std::vector<cv::cuda::GpuMat> normalized_channels;
        normalized_channels.resize(channels.size());
        for (size_t i = 0; i < channels.size(); ++i)
        {
            cv::cuda::normalize(channels[i],
                                normalized_channels,
                                alpha,
                                beta,
                                norm_type.getValue(),
                                input_image->getDepth(),
                                mask == NULL ? cv::noArray() : mask->getGpuMat(stream()),
                                stream());
        }
        if (input_image->getNumMats() == 1)
        {
            cv::cuda::merge(channels, normalized, stream());
            normalized_output_param.updateData(normalized, input_image_param.getTimestamp(), _ctx.get());
        }
        else
        {
            normalized_output_param.updateData(normalized_channels, input_image_param.getTimestamp(), _ctx.get());
        }
        return true;
    }
}

MO_REGISTER_CLASS(AutoScale)
MO_REGISTER_CLASS(Normalize)
MO_REGISTER_CLASS(Scale)
#endif
