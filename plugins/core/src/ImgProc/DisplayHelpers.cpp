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

std::string DrawDetectionsBase::textLabel(const DetectedObject& det)
{
    std::stringstream ss;

    if (draw_class_label)
    {
        if (det.classifications.size())
        {
            if (det.classifications[0].cat)
                ss << det.classifications[0].cat->getName() << ":";
            ss << std::setprecision(3) << det.classifications[0].conf;
        }
    }

    if (draw_detection_id)
        ss << " - " << det.id;
    return ss.str();
}

bool DrawDetections::processImpl()
{
    cv::cuda::GpuMat device_draw_image;
    cv::Mat host_draw_image;
    cv::Size size;
    if (_ctx->device_id != -1)
    {
        image->clone(device_draw_image, stream());
        size = device_draw_image.size();
    }
    else
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
            if (!device_draw_image.empty())
                cv::cuda::rectangle(device_draw_image, rect, color, 3, stream());
            else
                cv::rectangle(host_draw_image, rect.tl(), rect.br(), color, 3);

            if (detection.classifications.size())
            {
                if (detection.classifications[0].cat)
                {
                    color = detection.classifications[0].cat->color;
                }
            }
            cv::Rect text_rect = cv::Rect(rect.tl() + cv::Point(10, 20), cv::Size(200, 20));
            if ((cv::Rect({0, 0}, size) & text_rect) == text_rect)
            {
                cv::Mat text_image(20, 200, CV_8UC3);
                text_image.setTo(cv::Scalar::all(0));
                cv::putText(text_image, textLabel(detection), {0, 15}, cv::FONT_HERSHEY_COMPLEX, 0.4, color);
                if (!device_draw_image.empty())
                {
                    cv::cuda::GpuMat d_text;
                    d_text.upload(text_image, stream());
                    cv::cuda::GpuMat text_roi = device_draw_image(text_rect);
                    cv::cuda::add(text_roi, d_text, text_roi, cv::noArray(), -1, stream());
                    drawn_text.push_back(text_image); // need to prevent recycling of the images too early
                }
                else
                {
                    auto text_roi = host_draw_image(text_rect);
                    cv::add(text_roi, text_image, text_roi);
                }
            }
        }
    }
    if (device_draw_image.empty())
    {
        output_param.updateData(host_draw_image, mo::tag::_param = image_param, _ctx.get());
    }
    else
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

bool DrawDescriptors::processImpl()
{
    cv::cuda::GpuMat device_draw_image;
    cv::Mat host_draw_image;
    cv::Size size;
    if (_ctx->device_id == -1)
    {
        image->clone(host_draw_image);
        size = host_draw_image.size();
    }
    else
    {
        if (image->getSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            if (image->clone(host_draw_image, stream()))
            {
                stream().waitForCompletion();
            }
            size = host_draw_image.size();
        }
        else
        {
            image->clone(device_draw_image, stream());
            size = device_draw_image.size();
        }
    }
    const bool draw_gpu = !device_draw_image.empty();
    std::vector<cv::Mat> drawn_text;
    for (const auto& det : *detections)
    {
        const cv::Rect rect = det.bounding_box;

        if (draw_gpu)
        {
        }
        else
        {
            cv::Mat descriptor;
            if (_ctx->device_id == -1)
            {
                descriptor = det.descriptor.getMatNoSync();
            }
            else
            {
                bool sync = false;
                descriptor = det.descriptor.getMat(stream(), 0, &sync);
                if (sync)
                {
                    stream().waitForCompletion();
                }
            }
            cv::normalize(descriptor, descriptor, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(descriptor, descriptor, cv::COLORMAP_HSV);
            const int height = static_cast<int>(rect.height);
            const int width = descriptor.cols / height;
            cv::Point tl = det.bounding_box.tl();
            tl.x += det.bounding_box.width + 5;
            if (tl.x + width >= host_draw_image.cols)
            {
                tl.x = rect.x - width - 5;
            }
            for (int i = 0; i < descriptor.cols; ++i)
            {
                host_draw_image.at<cv::Vec3b>(i % height + tl.y, tl.x + i / descriptor.cols) =
                    descriptor.at<cv::Vec3b>(i);
            }
            cv::Scalar color;
            if (det.classifications.size())
            {
                if (det.classifications[0].cat)
                {
                    color = det.classifications[0].cat->color;
                }
            }
            cv::rectangle(host_draw_image, rect.tl(), rect.br(), color, 3);

            cv::Rect text_rect = cv::Rect(rect.tl() + cv::Point(10, 20), cv::Size(200, 20));
            if ((cv::Rect({0, 0}, size) & text_rect) == text_rect)
            {
                cv::Mat text_image(20, 200, CV_8UC3);
                text_image.setTo(cv::Scalar::all(0));
                cv::putText(text_image, textLabel(det), {0, 15}, cv::FONT_HERSHEY_COMPLEX, 0.4, color);
                if (!device_draw_image.empty())
                {
                    cv::cuda::GpuMat d_text;
                    d_text.upload(text_image, stream());
                    cv::cuda::GpuMat text_roi = device_draw_image(text_rect);
                    cv::cuda::add(text_roi, d_text, text_roi, cv::noArray(), -1, stream());
                    drawn_text.push_back(text_image); // need to prevent recycling of the images too early
                }
                else
                {
                    auto text_roi = host_draw_image(text_rect);
                    cv::add(text_roi, text_image, text_roi);
                }
            }
        }
    }
    if (device_draw_image.empty())
    {
        output_param.updateData(host_draw_image, mo::tag::_param = image_param, mo::tag::_context = _ctx.get());
    }
    else
    {
    }

    return true;
}

MO_REGISTER_CLASS(AutoScale)
MO_REGISTER_CLASS(Normalize)
MO_REGISTER_CLASS(DrawDetections)
MO_REGISTER_CLASS(DrawDescriptors)
