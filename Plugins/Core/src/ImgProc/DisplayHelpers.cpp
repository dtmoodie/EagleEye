#include "DisplayHelpers.h"
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include <Aquila/utilities/GpuDrawing.hpp>
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
#include <fstream>

using namespace aq;
using namespace aq::Nodes;

bool Scale::ProcessImpl()
{
    cv::cuda::GpuMat scaled;
    cv::cuda::multiply(input->GetGpuMat(Stream()), cv::Scalar(scale_factor), scaled, 1, -1, Stream());
    output_param.UpdateData(scaled, input_param.GetTimestamp(), _ctx);
    return true;
}
MO_REGISTER_CLASS(Scale)

bool AutoScale::ProcessImpl()
{
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(input_image->GetGpuMat(Stream()), channels, Stream());
    for(size_t i = 0; i < channels.size(); ++i)
    {
        double minVal, maxVal;
        cv::cuda::minMax(channels[i], &minVal, &maxVal);
        double scaleFactor = 255.0 / (maxVal - minVal);
        channels[i].convertTo(channels[0], CV_8U, scaleFactor, minVal*scaleFactor);
        UpdateParameter<double>("Min-" + boost::lexical_cast<std::string>(i), minVal)->SetFlags(mo::State_e);
        UpdateParameter<double>("Max-" + boost::lexical_cast<std::string>(i), maxVal)->SetFlags(mo::State_e);
    }
    cv::cuda::merge(channels,output_image.GetGpuMat(Stream()), Stream());
    return true;
}

bool DrawDetections::ProcessImpl()
{
    if(colors.size() != labels->size())
    {
        colors.resize(labels->size());
        for(int i = 0; i < colors.size(); ++i)
        {
            colors[i] = cv::Vec3b(i * 180 / colors.size(), 200, 255);
        }
        cv::Mat colors_mat(colors.size(), 1, CV_8UC3, &colors[0]);
        cv::cvtColor(colors_mat, colors_mat, cv::COLOR_HSV2BGR);
    }
    cv::cuda::GpuMat draw_image;
    image->Clone(draw_image, Stream());
    std::vector<cv::Mat> drawn_text;

    if(detections)
    {
        for(auto& detection : *detections)
        {
            cv::Rect rect(detection.boundingBox.x, detection.boundingBox.y, detection.boundingBox.width, detection.boundingBox.height);
            cv::Scalar color;
            std::stringstream ss;
            if(labels->size())
            {
                color = colors[detection.classification.classNumber];
                if(detection.classification.classNumber > 0 && detection.classification.classNumber < labels->size())
                {
                    ss << (*labels)[detection.classification.classNumber] << " : " << std::setprecision(3) << detection.classification.confidence;
                }else
                {
                    ss << std::setprecision(3) << detection.classification.confidence;
                }
                ss << " - " << detection.id;
            }else
            {
                // random color for each different detection
                if(detection.classification.classNumber >= colors.size())
                {
                    colors.resize(detection.classification.classNumber + 1);
                    for(int i = 0; i < colors.size(); ++i)
                    {
                        colors[i] = cv::Vec3b(i * 180 / colors.size(), 200, 255);
                    }
                    cv::Mat colors_mat(colors.size(), 1, CV_8UC3, &colors[0]);
                    cv::cvtColor(colors_mat, colors_mat, cv::COLOR_HSV2BGR);
                }
                color = colors[detection.classification.classNumber];
            }
            cv::cuda::rectangle(draw_image, rect, color, 3, Stream());

            cv::Rect text_rect = cv::Rect(rect.tl() + cv::Point(10,20), cv::Size(200,20));
            if((cv::Rect({0,0}, draw_image.size()) & text_rect) == text_rect)
            {
                cv::Mat text_image(20, 200, CV_8UC3);
                text_image.setTo(cv::Scalar::all(0));
                cv::putText(text_image, ss.str(), {0, 15}, cv::FONT_HERSHEY_COMPLEX, 0.4, color);
                cv::cuda::GpuMat d_text;
                d_text.upload(text_image, Stream());
                cv::cuda::GpuMat text_roi = draw_image(text_rect);
                cv::cuda::add(text_roi, d_text, text_roi, cv::noArray(), -1, Stream());
                drawn_text.push_back(text_image); // need to prevent recycling of the images too early
            }

        }
    }
    image_with_detections_param.UpdateData(draw_image, image_param.GetTimestamp(), _ctx);
    return true;
}

bool Normalize::ProcessImpl()
{
    cv::cuda::GpuMat normalized;

    if(input_image->GetChannels() == 1)
    {
        cv::cuda::normalize(input_image->GetGpuMat(Stream()),
            normalized,
            alpha,
            beta,
            norm_type.currentSelection, input_image->GetDepth(),
            mask == NULL ? cv::noArray(): mask->GetGpuMat(Stream()),
            Stream());
        normalized_output_param.UpdateData(normalized, input_image_param.GetTimestamp(), _ctx);
        return true;
    }else
    {
        std::vector<cv::cuda::GpuMat> channels;

        if (input_image->GetNumMats() == 1)
        {
            cv::cuda::split(input_image->GetGpuMat(Stream()), channels, Stream());
        }else
        {
            channels = input_image->GetGpuMatVec(Stream());
        }
        std::vector<cv::cuda::GpuMat> normalized_channels;
        normalized_channels.resize(channels.size());
        for(int i = 0; i < channels.size(); ++i)
        {
            cv::cuda::normalize(channels[i], normalized_channels,
                alpha,
                beta,
                norm_type.getValue(), input_image->GetDepth(),
                mask == NULL ? cv::noArray() : mask->GetGpuMat(Stream()),
                Stream());
        }
        if(input_image->GetNumMats() == 1)
        {
            cv::cuda::merge(channels, normalized, Stream());
            normalized_output_param.UpdateData(normalized, input_image_param.GetTimestamp(), _ctx);
        }else
        {
            normalized_output_param.UpdateData(normalized_channels, input_image_param.GetTimestamp(), _ctx);
        }
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(AutoScale)
MO_REGISTER_CLASS(Normalize)
MO_REGISTER_CLASS(DrawDetections)

