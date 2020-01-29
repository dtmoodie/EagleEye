#include "DrawDetections.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/utilities/GpuDrawing.hpp>
#include <MetaObject/params/TypeSelector.hpp>
#include <iomanip>

namespace aq
{
namespace nodes
{

void DrawDetections::drawMetaData(cv::Mat&, const aq::DetectedObject&, const cv::Rect2f&, const size_t)
{
}

void DrawDetections::drawMetaData(cv::Mat& host_draw_image,
                                  const aq::DetectionDescription& det,
                                  const cv::Rect2f& rect,
                                  const size_t idx)
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
    cv::Point tl = rect.tl();
    tl.x += rect.width + 5;
    if (tl.x + width >= host_draw_image.cols)
    {
        tl.x = rect.x - width - 5;
    }
    if (tl.y < 0)
    {
        tl.y = 0;
    }
    for (int i = 0; i < descriptor.cols; ++i)
    {
        host_draw_image.at<cv::Vec3b>(i % height + tl.y, tl.x + i / descriptor.cols) = descriptor.at<cv::Vec3b>(i);
    }
}

void DrawDetections::drawMetaData(cv::Mat& mat,
                                  const aq::DetectionDescriptionPatch& det,
                                  const cv::Rect2f& rect,
                                  const size_t idx)
{
    drawMetaData(mat, static_cast<const aq::DetectionDescription&>(det), rect, idx);
}

template <class DType>
void DrawDetections::apply(bool* success)
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
    const bool draw_gpu = !device_draw_image.empty();
    std::vector<cv::Mat> drawn_text;
    auto det_ts = detections_param.getInputTimestamp();
    if (det_ts != image_param.getInputTimestamp())
    {
        *success = true;
        return;
    }
    const DType* dets = mo::get<const DType*>(detections);

    for (size_t i = 0; i < dets->size(); ++i)
    {
        auto detection = (*dets)[i];
        boundingBoxToPixels(detection.bounding_box, size);
        cv::Rect rect(static_cast<int>(detection.bounding_box.x),
                      static_cast<int>(detection.bounding_box.y),
                      static_cast<int>(detection.bounding_box.width),
                      static_cast<int>(detection.bounding_box.height));
        cv::Scalar color;
        if (detection.classifications.size())
        {
            if (detection.classifications[0].cat)
            {
                color = detection.classifications[0].cat->color;
            }
        }

        if (draw_gpu)
        {
            cv::cuda::rectangle(device_draw_image, rect, color, 3, stream());
        }
        else
        {
            cv::rectangle(host_draw_image, rect.tl(), rect.br(), color, 3);
            drawMetaData(host_draw_image, detection, rect, i);
        }

        cv::Rect text_rect = cv::Rect(rect.tl() + cv::Point(10, 20), cv::Size(200, 20));
        if ((cv::Rect({0, 0}, size) & text_rect) == text_rect)
        {
            cv::Mat text_image(20, 200, CV_8UC3);
            text_image.setTo(cv::Scalar::all(0));
            cv::putText(text_image, textLabel(detection), {0, 15}, cv::FONT_HERSHEY_COMPLEX, 0.4, color);
            if (draw_gpu)
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
    if (publish_empty_dets || !dets->empty())
    {
        if (device_draw_image.empty())
        {
            output_param.updateData(host_draw_image, mo::tag::_param = image_param, _ctx.get());
        }
        else
        {
            output_param.updateData(device_draw_image, mo::tag::_param = image_param, _ctx.get());
        }
    }

    *success = true;
    return;
}

std::string DrawDetections::textLabel(const DetectedObject& det)
{
    std::stringstream ss;

    if (draw_class_label)
    {
        if (det.classifications.size())
        {
            if (det.classifications[0].cat)
            {
                ss << det.classifications[0].cat->getName() << ":";
            }
            ss << std::setprecision(3) << det.classifications[0].conf;
        }
    }

    if (draw_detection_id)
    {
        ss << " - " << det.id;
    }
    return ss.str();
}

bool DrawDetections::processImpl()
{
    bool success = false;
    mo::selectType<decltype(detections_param)::TypeTuple>(*this, detections_param.getTypeInfo(), &success);
    return success;
}
}
}

using namespace aq::nodes;

MO_REGISTER_CLASS(DrawDetections)
