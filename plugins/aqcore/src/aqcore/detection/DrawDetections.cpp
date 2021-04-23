#include <ct/types/opencv.hpp>

#include <Aquila/types/SyncedMemory.hpp>

#include "DrawDetections.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <MetaObject/params/TypeSelector.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iomanip>

namespace aqcore
{
    cv::Scalar getColor(const aq::detection::Classifications& cat)
    {
        cv::Scalar color = cv::Scalar::all(0);
        if (cat.size() && cat[0].cat)
        {
            color = cat[0].cat->color;
        }
        return color;
    }
    void DrawDetections::drawBoxes(cv::Mat& mat,
                                   mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                                   mt::Tensor<const Classifications, 1> cats)
    {
        for (size_t i = 0; i < bbs.getShape()[0]; ++i)
        {
            const Classifications& cat = cats[i];
            const cv::Rect bb = bbs[i];
            const cv::Scalar color = getColor(cat);
            constexpr const uint32_t thickness = 3U;
            cv::rectangle(mat, bb.tl(), bb.br(), color, thickness);
        }
    }

    void DrawDetections::drawBoxes(cv::cuda::GpuMat& mat,
                                   mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                                   mt::Tensor<const Classifications, 1> cls,
                                   cv::cuda::Stream& stream)
    {
    }

    void DrawDetections::drawLabels(cv::Mat& mat,
                                    mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                                    mt::Tensor<const Classifications, 1> cats,
                                    mt::Tensor<const Id::DType, 1> ids)
    {
        const cv::Size size = mat.size();
        const cv::Rect img_rect({0, 0}, size);
        const uint32_t num_dets = bbs.getShape()[0];
        for (size_t i = 0; i < num_dets; ++i)
        {
            const cv::Rect2f bb = bbs[i];
            const cv::Point tl = cv::Point(bb.tl()) + cv::Point(10, 20);
            const cv::Rect text_rect(tl, cv::Size(200, 20));
            if ((img_rect & text_rect) == text_rect)
            {
                cv::Mat text_img = textImage(cats[i], ids[i]);
                cv::Mat text_roi = mat(text_rect);
                cv::add(text_roi, text_img, text_roi);
            }
        }
    }

    void DrawDetections::drawLabels(cv::cuda::GpuMat& mat,
                                    mt::Tensor<const typename BoundingBox2d::DType, 1> bbs,
                                    mt::Tensor<const Classifications, 1> cats,
                                    mt::Tensor<const typename Id::DType, 1> ids,
                                    cv::cuda::Stream& stream)
    {
    }

    void DrawDetections::drawMetaData(cv::Mat mat,
                                      mt::Tensor<const typename BoundingBox2d::DType, 1> bbs,
                                      mt::Tensor<const float, 2> descriptors)
    {
        const auto num_detections = bbs.getShape()[0];
        for (size_t i = 0; i < num_detections; ++i) {}
    }

    void DrawDetections::drawMetaData(cv::cuda::GpuMat& mat,
                                      mt::Tensor<const typename BoundingBox2d::DType, 1> bbs,
                                      mt::Tensor<const float, 2> descriptors,
                                      cv::cuda::Stream& stream)
    {
    }

    /*void DrawDetections::drawMetaData(cv::Mat& host_draw_image,
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
    }*/

    /*void DrawDetections::drawMetaData(cv::Mat& mat,
                                      const aq::DetectionDescriptionPatch& det,
                                      const cv::Rect2f& rect,
                                      const size_t idx)
    {
        drawMetaData(mat, static_cast<const aq::DetectionDescription&>(det), rect, idx);
    }*/

    std::string DrawDetections::textLabel(const Classifications& cat, const Id& id)
    {
        std::stringstream ss;

        if (draw_class_label)
        {
            if (cat.size())
            {
                if (cat[0].cat)
                {
                    ss << cat[0].cat->getName() << ":";
                }
                ss << std::setprecision(3) << cat[0].conf;
            }
        }

        if (draw_detection_id)
        {
            ss << " - " << id.m_value;
        }
        return std::move(ss).str();
    }

    cv::Mat DrawDetections::textImage(const Classifications& cat, const Id& id)
    {
        cv::Mat text_image(20, 200, CV_8UC3);
        text_image.setTo(cv::Scalar::all(0));
        std::string text = textLabel(cat, id);
        const cv::Scalar color = getColor(cat);
        cv::putText(text_image, text, {0, 15}, cv::FONT_HERSHEY_SIMPLEX, 0.4, color);
        return text_image;
    }

    bool DrawDetections::processImpl()
    {
        // Drawing on the CPU is very cheap, so only draw on the GPU if the image is already there
        const bool image_is_on_device = this->image->state() == aq::SyncedMemory::SyncState::DEVICE_UPDATED;
        const bool draw_on_device = this->getStream()->isDeviceStream() && image_is_on_device;

        cv::Size size;
        mt::Tensor<const typename BoundingBox2d::DType, 1> bbs = detections->getComponent<BoundingBox2d>();
        mt::Tensor<const Classifications, 1> cats = detections->getComponent<Classifications>();
        mt::Tensor<const typename Id::DType, 1> ids = detections->getComponent<Id>();
        mt::Tensor<const float, 2> descriptors = detections->getComponent<Descriptor>();

        if (draw_on_device)
        {
            cv::cuda::GpuMat device_draw_image;

            this->image->copyTo(device_draw_image, this->getStream()->getDeviceStream());

            auto& stream = this->getCVStream();
            drawBoxes(device_draw_image, bbs, cats, stream);
            drawLabels(device_draw_image, bbs, cats, ids, stream);
            drawMetaData(device_draw_image, bbs, descriptors, stream);
        }
        else
        {
            cv::Mat host_draw_image;
            image->copyTo(host_draw_image);
            drawBoxes(host_draw_image, bbs, cats);
            drawLabels(host_draw_image, bbs, cats, ids);
            drawMetaData(host_draw_image, bbs, descriptors);
            this->output.publish(std::move(host_draw_image), mo::tags::param = &this->image_param);
        }

        /*if (_ctx->device_id != -1)
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

        *success = true;*/
        return true;
    }

} // namespace aqcore

using namespace aqcore;

MO_REGISTER_CLASS(DrawDetections)
