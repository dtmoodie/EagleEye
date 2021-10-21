#include <ct/types/opencv.hpp>

#include <Aquila/types/SyncedMemory.hpp>

#include "DetectionOverlay.hpp"
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>

namespace aqcore
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

    void DetectionOverlay::drawOverlay(mo::IDeviceStream& stream)
    {
        const auto state = image->state();
        if (state < state.DEVICE_UPDATED)
        {
            drawOverlay(static_cast<mo::IAsyncStream&>(stream));
            return;
        }
        cv::cuda::GpuMat im;
        cv::cuda::Stream& cv_stream = this->getCVStream();
        image->copyTo(im, &stream);
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
                        const auto roi = patch_itr->second.patch.gpuMat(&stream);
                        auto dest_roi = im(cv::Rect(i * width, im.rows - width - y_offset, width, width));
                        if (roi.size() == dest_roi.size())
                        {
                            roi.copyTo(dest_roi, cv_stream);
                        }
                        else
                        {
                            cv::cuda::resize(roi, dest_roi, dest_roi.size(), 0, 0, cv::INTER_NEAREST, cv_stream);
                        }

                        auto draw_text = [&draw_mats, &y_offset, &cv_stream, &im, width, i](cv::Mat img) {
                            auto dest_roi = im(cv::Rect(i * width, im.rows - width - y_offset, width, text_height));
                            dest_roi.upload(img, cv_stream);
                            draw_mats.push_back(std::move(img));
                            y_offset += text_height;
                        };
                        y_offset += text_height;
                        if (draw_conf)
                        {
                            auto color = patch_itr->second.color;
                            dest_roi.colRange(0, 3).setTo(cv::Scalar::all(0), cv_stream);
                            uint32_t conf_rows = width * patch_itr->second.det_conf;
                            conf_rows = std::max<uint32_t>(conf_rows, 1);
                            conf_rows = std::min<uint32_t>(conf_rows, width);
                            dest_roi.colRange(0, 3).rowRange(width - conf_rows, width).setTo(color, cv_stream);

                            conf_rows = width * patch_itr->second.cat_conf;
                            conf_rows = std::max<uint32_t>(conf_rows, 1);
                            conf_rows = std::min<uint32_t>(conf_rows, width);
                            dest_roi.colRange(width - 3, width).setTo(cv::Scalar::all(0), cv_stream);
                            dest_roi.colRange(width - 3, width)
                                .rowRange(width - conf_rows, width)
                                .setTo(color, cv_stream);

                            if (draw_classification)
                            {
                                auto name = drawText(patch_itr->second.classification, color, width);
                                draw_text(name);
                            }
                        }
                        if (draw_age)
                        {
                            auto header = image_param.getNewestHeader();
                            if (header && header->timestamp)
                            {
                                const mo::Time delta(*header->timestamp - patch_itr->second.last_seen_time);
                                auto time_string = delta.print(false, false, true, true, false);
                                auto age = drawText(std::move(time_string), cv::Scalar(0, 255, 0), width);
                                draw_text(age);
                            }
                        }
                        if (draw_timestamp)
                        {
                            auto time_string = patch_itr->second.last_seen_time.print(false, true, true, false, false);
                            auto age = drawText(time_string, cv::Scalar(0, 255, 0), width);
                            draw_text(age);
                        }
                    }
                }
            }
            stream.pushWork([draw_mats](mo::IAsyncStream*) {});
        }

        output.publish(aq::SyncedImage(im), mo::tags::param = &image_param);
    }
} // namespace aqcore
