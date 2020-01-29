#include "DetectionOverlay.hpp"
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include <Aquila/utilities/cuda/CudaCallbacks.hpp>
#include <MetaObject/core/CvContext.hpp>
namespace aq
{
namespace nodes
{
const int text_height = 15;

cv::Mat drawText(const std::string& text, cv::Scalar color, int width)
{
    cv::Mat name;
    name.create(text_height, width, CV_8UC3);
    name.setTo(cv::Scalar::all(0));
    cv::putText(name, text, {0, text_height - 4}, cv::FONT_HERSHEY_COMPLEX, 0.4, color, 1);
    return name;
}

template <>
void DetectionOverlay::drawOverlay(mo::CvContext* ctx)
{
    if (image->getSyncState() < image->DEVICE_UPDATED)
    {
        drawOverlay(static_cast<mo::Context*>(ctx));
        return;
    }
    cv::cuda::GpuMat im;
    auto stream = ctx->getStream();
    image->clone(im, stream);
    if (max_num_tiles > 0)
    {
        const uint32_t width = static_cast<uint32_t>(im.cols) / max_num_tiles;
        std::vector<cv::Mat> draw_mats;

        for (size_t i = 0; i < m_draw_locations.size(); ++i)
        {
            const auto patch_itr = m_detection_patches.find(m_draw_locations[i]);
            if (patch_itr != m_detection_patches.end())
            {
                int y_offset = 2;
                if ((i * width + width) < im.cols)
                {
                    const auto roi = patch_itr->second.patch.getGpuMat(ctx);
                    auto dest_roi = im(cv::Rect(i * width, im.rows - width - y_offset, width, width));
                    if (roi.size() == dest_roi.size())
                    {
                        roi.copyTo(dest_roi, stream);
                    }
                    else
                    {
                        cv::cuda::resize(roi, dest_roi, dest_roi.size(), 0, 0, cv::INTER_NEAREST, stream);
                    }

                    auto draw_text = [&draw_mats, &y_offset, &stream, &im, width, i](cv::Mat img) {
                        auto dest_roi = im(cv::Rect(i * width, im.rows - width - y_offset, width, text_height));
                        dest_roi.upload(img, stream);
                        draw_mats.push_back(std::move(img));
                        y_offset += text_height;
                    };
                    y_offset += text_height;
                    if (draw_conf)
                    {
                        auto color = patch_itr->second.color;
                        dest_roi.colRange(0, 3).setTo(cv::Scalar::all(0), stream);
                        uint32_t conf_rows = width * patch_itr->second.det_conf;
                        conf_rows = std::max<uint32_t>(conf_rows, 1);
                        conf_rows = std::min<uint32_t>(conf_rows, width);
                        dest_roi.colRange(0, 3).rowRange(width - conf_rows, width).setTo(color, stream);

                        conf_rows = width * patch_itr->second.cat_conf;
                        conf_rows = std::max<uint32_t>(conf_rows, 1);
                        conf_rows = std::min<uint32_t>(conf_rows, width);
                        dest_roi.colRange(width - 3, width).setTo(cv::Scalar::all(0), stream);
                        dest_roi.colRange(width - 3, width).rowRange(width - conf_rows, width).setTo(color, stream);

                        if (draw_classification)
                        {
                            auto name = drawText(patch_itr->second.classification, color, width);
                            draw_text(name);
                        }
                    }
                    if (draw_age)
                    {

                        auto now = image_param.getTimestamp();
                        if (now)
                        {

                            auto age = drawText(
                                mo::printTime(
                                    (*now - patch_itr->second.last_seen_time), false, false, true, true, false),
                                cv::Scalar(0, 255, 0),
                                width);
                            draw_text(age);
                        }
                    }
                    if (draw_timestamp)
                    {
                        auto age =
                            drawText(mo::printTime(patch_itr->second.last_seen_time, false, true, true, false, false),
                                     cv::Scalar(0, 255, 0),
                                     width);
                        draw_text(age);
                    }
                }
            }
        }
        aq::cuda::enqueue_callback_async([draw_mats]() {}, _ctx->thread_id, stream);
    }

    output_param.updateData(im, mo::tag::_param = image_param);
}
}
}
