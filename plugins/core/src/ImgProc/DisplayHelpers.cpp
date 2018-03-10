#include "DisplayHelpers.h"

#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <Aquila/rcc/external_includes/aqcore_link_libs.hpp>
#include <Aquila/utilities/GpuDrawing.hpp>
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
MO_REGISTER_CLASS(Scale)

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

bool DrawDetections::processImpl()
{
    cv::cuda::GpuMat device_draw_image;
    cv::Mat host_draw_image;
    cv::Size size;
    if(_ctx->device_id != -1)
    {
        image->clone(device_draw_image, stream());
        size = device_draw_image.size();
    }else
    {
        image->clone(host_draw_image);
        size = host_draw_image.size();
    }
    std::vector<cv::Mat> drawn_text;
    auto det_ts = detections_param.getTimestamp();
    if (det_ts != image_param.getTimestamp())
    {
        return true;
    }

    if (detections)
    {
        for (auto& detection : *detections)
        {
            cv::Rect rect(static_cast<int>(detection.bounding_box.x),
                          static_cast<int>(detection.bounding_box.y),
                          static_cast<int>(detection.bounding_box.width),
                          static_cast<int>(detection.bounding_box.height));
            cv::Scalar color;
            std::stringstream ss;
            if (detection.classifications.size())
            {
                if (detection.classifications[0].cat)
                {
                    color = detection.classifications[0].cat->color;
                }
            }

            if (draw_class_label)
            {
                if (detection.classifications[0].cat)
                    ss << detection.classifications[0].cat->getName() << ":";
                ss << std::setprecision(3) << detection.classifications[0].conf;
            }

            if (draw_detection_id)

                if (draw_detection_id)
                    ss << " - " << detection.id;
            if(!device_draw_image.empty())
                cv::cuda::rectangle(device_draw_image, rect, color, 3, stream());
            else
                cv::rectangle(host_draw_image, rect.tl(), rect.br(), color, 3);

            cv::Rect text_rect = cv::Rect(rect.tl() + cv::Point(10, 20), cv::Size(200, 20));
            if ((cv::Rect({0, 0}, size) & text_rect) == text_rect)
            {
                cv::Mat text_image(20, 200, CV_8UC3);
                text_image.setTo(cv::Scalar::all(0));
                cv::putText(text_image, ss.str(), {0, 15}, cv::FONT_HERSHEY_COMPLEX, 0.4, color);
                if(!device_draw_image.empty())
                {
                    cv::cuda::GpuMat d_text;
                    d_text.upload(text_image, stream());
                    cv::cuda::GpuMat text_roi = device_draw_image(text_rect);
                    cv::cuda::add(text_roi, d_text, text_roi, cv::noArray(), -1, stream());
                    drawn_text.push_back(text_image); // need to prevent recycling of the images too early
                }else
                {
                    auto text_roi = host_draw_image(text_rect);
                    cv::add(text_roi, text_image, text_roi);
                }
            }
        }
    }
    if(device_draw_image.empty())
    {
        output_param.updateData(host_draw_image, mo::tag::_param = image_param, _ctx.get());
    }else
    {
        output_param.updateData(device_draw_image, mo::tag::_param = image_param, _ctx.get());
    }
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
MO_REGISTER_CLASS(DrawDetections)
