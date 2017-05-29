#include "DisplayHelpers.h"
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <Aquila/utilities/GpuDrawing.hpp>
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <fstream>

using namespace aq;
using namespace aq::Nodes;

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
    for(size_t i = 0; i < channels.size(); ++i)
    {
        double minVal, maxVal;
        cv::cuda::minMax(channels[i], &minVal, &maxVal);
        double scaleFactor = 255.0 / (maxVal - minVal);
        channels[i].convertTo(channels[0], CV_8U, scaleFactor, minVal*scaleFactor);
        // TODO update to new api
        //updateParameter<double>("Min-" + boost::lexical_cast<std::string>(i), minVal)->setFlags(mo::State_e);
        //updateParameter<double>("Max-" + boost::lexical_cast<std::string>(i), maxVal)->setFlags(mo::State_e);
    }
    cv::cuda::merge(channels,output_image.getGpuMat(stream()), stream());
    return true;
}

bool DrawDetections::processImpl()
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
    image->clone(draw_image, stream());
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
                    if(draw_class_label)
                        ss << (*labels)[detection.classification.classNumber] << " : " << std::setprecision(3) << detection.classification.confidence;
                }else
                {
                    if(draw_class_label)
                        ss << std::setprecision(3) << detection.classification.confidence;
                }
                if(draw_detection_id)
                    ss << " - ";
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
            if(draw_detection_id)
                ss << detection.id;
            cv::cuda::rectangle(draw_image, rect, color, 3, stream());

            cv::Rect text_rect = cv::Rect(rect.tl() + cv::Point(10,20), cv::Size(200,20));
            if((cv::Rect({0,0}, draw_image.size()) & text_rect) == text_rect)
            {
                cv::Mat text_image(20, 200, CV_8UC3);
                text_image.setTo(cv::Scalar::all(0));
                cv::putText(text_image, ss.str(), {0, 15}, cv::FONT_HERSHEY_COMPLEX, 0.4, color);
                cv::cuda::GpuMat d_text;
                d_text.upload(text_image, stream());
                cv::cuda::GpuMat text_roi = draw_image(text_rect);
                cv::cuda::add(text_roi, d_text, text_roi, cv::noArray(), -1, stream());
                drawn_text.push_back(text_image); // need to prevent recycling of the images too early
            }

        }
    }
    image_with_detections_param.updateData(draw_image, image_param.getTimestamp(), _ctx.get());
    return true;
}

bool Normalize::processImpl()
{
    cv::cuda::GpuMat normalized;

    if(input_image->getChannels() == 1)
    {
        cv::cuda::normalize(input_image->getGpuMat(stream()),
            normalized,
            alpha,
            beta,
            norm_type.currentSelection, input_image->getDepth(),
            mask == NULL ? cv::noArray(): mask->getGpuMat(stream()),
            stream());
        normalized_output_param.updateData(normalized, input_image_param.getTimestamp(), _ctx.get());
        return true;
    }else
    {
        std::vector<cv::cuda::GpuMat> channels;

        if (input_image->getNumMats() == 1)
        {
            cv::cuda::split(input_image->getGpuMat(stream()), channels, stream());
        }else
        {
            channels = input_image->getGpuMatVec(stream());
        }
        std::vector<cv::cuda::GpuMat> normalized_channels;
        normalized_channels.resize(channels.size());
        for(int i = 0; i < channels.size(); ++i)
        {
            cv::cuda::normalize(channels[i], normalized_channels,
                alpha,
                beta,
                norm_type.getValue(), input_image->getDepth(),
                mask == NULL ? cv::noArray() : mask->getGpuMat(stream()),
                stream());
        }
        if(input_image->getNumMats() == 1)
        {
            cv::cuda::merge(channels, normalized, stream());
            normalized_output_param.updateData(normalized, input_image_param.getTimestamp(), _ctx.get());
        }else
        {
            normalized_output_param.updateData(normalized_channels, input_image_param.getTimestamp(), _ctx.get());
        }
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(AutoScale)
MO_REGISTER_CLASS(Normalize)
MO_REGISTER_CLASS(DrawDetections)

