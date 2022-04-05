#include <MetaObject/types/opencv.hpp>

#include <Aquila/types/SyncedMemory.hpp>

#include "DetectionOverlay.hpp"

#include <Aquila/types/CVStream.hpp>
#include <Aquila/types/DetectionPatch.hpp>

#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>

namespace aqcore
{

    template <class CTX>
    bool DetectionOverlay::processImpl(CTX& ctx)
    {
        mo::OptionalTime ts;
        auto header = detections_param.getNewestHeader();
        if (header)
        {
            ts = header->timestamp;
        }
        if (!ts)
        {
            ts = mo::Time::now();
        }

        this->pruneOldDetections(*ts);
        updateOverlay(*detections, *ts, ctx);
        drawOverlay(ctx);
        return true;
    }

    bool DetectionOverlay::processImpl()
    {
        auto stream = getStream();
        MO_ASSERT(stream);
        nodeStreamSwitch(this, *stream);
        return true;
    }

    void
    DetectionOverlay::updateOverlay(const aq::DetectedObjectSet& dets, const mo::Time& ts, mo::IAsyncStream& stream)
    {
        mt::Tensor<const aq::Classification, 2> classifications = dets.getComponent<aq::detection::Classifications>();
        mt::Tensor<const aq::detection::Confidence::DType, 1> confidence =
            dets.getComponent<aq::detection::Confidence>();
        mt::Tensor<const aq::detection::Id::DType, 1> id = dets.getComponent<aq::detection::Id>();
        mt::Tensor<const aq::detection::AlignedPatch, 1> patches = dets.getComponent<aq::detection::AlignedPatch>();
        mt::Tensor<const aq::detection::BoundingBox2d::DType, 1> bb = dets.getComponent<aq::detection::BoundingBox2d>();
        const uint32_t num_dets = dets.getNumEntities();

        MO_ASSERT_EQ(id.getShape()[0], num_dets);
        MO_ASSERT_EQ(id.getShape()[0], num_dets);

        auto device_stream = std::dynamic_pointer_cast<mo::IDeviceStream>(getStream());
        const aq::PixelFormat fmt = image->pixelFormat();

        for (uint32_t i = 0; i < num_dets; ++i)
        {
            float cat_conf = 0.0F;
            cv::Scalar color(0, 255, 0);
            std::string cat = "None";
            if (classifications.getShape()[0] != 0)
            {
                if (classifications[i].getShape()[0] > 0)
                {
                    cat_conf = classifications[i][0].conf;
                    if (classifications[i][0].cat)
                    {
                        color = classifications[i][0].cat->color;
                        cat = classifications[i][0].cat->getName();
                    }
                }
            }

            aq::SyncedImage tmp;
            aq::SyncedImage patch_source;
            const aq::SyncedImage* patch = &tmp;
            if (patches.getShape()[0] == num_dets)
            {
                patch = &patches[i].aligned_patch;
                patch_source = patches[i].aligned_patch;
            }
            else
            {
                MO_ASSERT_EQ(bb.getShape()[0], num_dets);
                patch_source = *image;
                if (stream.isDeviceStream())
                {
                    auto gpumat = image->gpuMat();
                    tmp = aq::SyncedImage(gpumat(bb[i]), fmt, device_stream);
                }
                else
                {
                    auto mat = image->mat();
                    tmp = aq::SyncedImage(mat(bb[i]), fmt, device_stream);
                }
            }

            addOrUpdate(*patch, id[i], ts, cat_conf, confidence[i], color, cat, patch_source, stream);
        }
    }

    int DetectionOverlay::getWidth() const
    {
        if (max_num_tiles > 0)
        {
            auto shape = image->shape();
            return static_cast<int32_t>(shape(1) / max_num_tiles);
        }
        else
        {
            return 100;
        }
    }

    void DetectionOverlay::addOrUpdate(const aq::SyncedImage& patch_,
                                       aq::detection::Id id,
                                       const mo::Time& ts,
                                       float cat_conf,
                                       aq::detection::Confidence det_conf,
                                       cv::Scalar color,
                                       const std::string& classification,
                                       const aq::SyncedImage& source,
                                       mo::IAsyncStream& stream)
    {
        const auto width = getWidth();
        aq::SyncedImage patch;
        const auto state = patch_.state();
        // Square patches
        const cv::Size size(width, width);
        const auto interpolation = cv::INTER_LINEAR;
        if (state < state.DEVICE_UPDATED)
        {
            cv::Mat resized;
            auto mat = patch_.mat();
            stream.synchronize();
            cv::resize(mat, resized, size, 0.0, 0.0, interpolation);
            patch = aq::SyncedImage(resized);
        }
        else
        {
            cv::cuda::GpuMat resized;
            cv::cuda::GpuMat tmp = patch_.gpuMat();
            cv::cuda::resize(tmp, resized, size, 0.0, 0.0, interpolation, this->getCVStream());
            // not sure if this is necessary
            stream.synchronize();
            patch = aq::SyncedImage(resized);
        }

        auto itr = m_detection_patches.find(id);
        if (itr == m_detection_patches.end())
        {
            RenderedDet det;
            det.patch = patch;
            det.first_seen_time = ts;
            det.last_seen_time = ts;
            det.cat_conf = cat_conf;
            det.det_conf = det_conf;
            det.color = color;
            det.classification = classification;
            det.patch_source = source;
            m_detection_patches[id] = std::move(det);
            m_draw_locations.push_back(id);
        }
        else
        {
            itr->second.patch = patch;
            itr->second.last_seen_time = ts;
            itr->second.cat_conf = cat_conf;
            itr->second.det_conf = det_conf;
            itr->second.color = color;
            itr->second.classification = classification;
        }
    }

    void DetectionOverlay::drawOverlay(mo::IAsyncStream& stream)
    {
        cv::Mat im;
        cv::Mat tmp = *image;
        // sync image?
        tmp.copyTo(im);

        const uint32_t width = static_cast<uint32_t>(im.cols) / max_num_tiles;
        for (size_t i = 0; i < m_draw_locations.size(); ++i)
        {
            if ((i * width + width) < im.cols)
            {
                const auto patch_itr = m_detection_patches.find(m_draw_locations[i]);
                if (patch_itr != m_detection_patches.end())
                {
                    const auto roi = patch_itr->second.patch.mat();
                    cv::Mat dest_roi = im(cv::Rect(im.rows - width - 2, i * width, width, width));
                    cv::resize(roi, dest_roi, cv::Size(width, width), 0, 0, cv::INTER_LINEAR);
                    if (draw_conf)
                    {
                        auto color = patch_itr->second.color;
                        dest_roi.colRange(0, 3).setTo(cv::Scalar::all(0));
                        uint32_t conf_rows = width * patch_itr->second.det_conf;
                        conf_rows = std::max<uint32_t>(conf_rows, 1);
                        conf_rows = std::min<uint32_t>(conf_rows, width);
                        dest_roi.colRange(0, 3).rowRange(width - conf_rows, width).setTo(color);

                        conf_rows = width * patch_itr->second.cat_conf;
                        conf_rows = std::max<uint32_t>(conf_rows, 1);
                        conf_rows = std::min<uint32_t>(conf_rows, width);
                        dest_roi.colRange(width - 3, width).setTo(cv::Scalar::all(0));
                        dest_roi.colRange(width - 3, width).rowRange(width - conf_rows, width).setTo(color);
                    }
                }
            }
        }
        output.publish(im, mo::tags::param = &image_param);
    }

    void DetectionOverlay::pruneOldDetections(const mo::Time& ts)
    {
        // const auto size = image->getSize();
        for (auto itr = m_draw_locations.begin(); itr != m_draw_locations.end();)
        {
            auto patch_itr = m_detection_patches.find(*itr);
            if (patch_itr != m_detection_patches.end())
            {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(ts - patch_itr->second.last_seen_time)
                        .count() > (max_age_seconds * 1000.0F))
                {
                    m_detection_patches.erase(patch_itr);
                    itr = m_draw_locations.erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
            else
            {
                itr = m_draw_locations.erase(itr);
            }
        }
    }

} // namespace aqcore

using namespace aqcore;
MO_REGISTER_CLASS(DetectionOverlay)
