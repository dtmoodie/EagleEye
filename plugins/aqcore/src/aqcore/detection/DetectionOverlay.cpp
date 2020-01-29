#include "DetectionOverlay.hpp"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudawarping.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>

namespace aq
{
namespace nodes
{

template <class DetType, class CTX>
void DetectionOverlay::apply(CTX* ctx)
{
    const DetType* dets = mo::get<const DetType*>(detections);
    auto ts = detections_param.getInputTimestamp();
    if (!ts)
    {
        ts = mo::getCurrentTime();
    }
    pruneOldDetections(*ts);
    updateOverlay(*dets, *ts);
    drawOverlay(ctx);
}

template <class CTX>
bool DetectionOverlay::processImpl(CTX* ctx)
{
    return mo::selectType<decltype(detections_param)::TypeTuple>(*this, detections_param.getTypeInfo(), ctx);
}

bool DetectionOverlay::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

void DetectionOverlay::updateOverlay(const aq::DetectionDescriptionPatchSet& dets, const mo::Time_t& ts)
{
    for (const auto& det : dets)
    {
        float cat_conf = 0.0F;
        cv::Scalar color(0, 255, 0);
        std::string cat = "None";
        if (det.classifications.size() > 0)
        {
            cat_conf = det.classifications[0].conf;
            if (det.classifications[0].cat)
            {
                color = det.classifications[0].cat->color;
                cat = det.classifications[0].cat->getName();
            }
        }
        addOrUpdate(det.aligned_patch, det.id, ts, cat_conf, det.confidence, color, cat);
    }
}

template <class DetType>
void DetectionOverlay::updateOverlay(const DetType& dets, const mo::Time_t& ts)
{
    for (const auto& det : dets)
    {
        float cat_conf = 0.0F;
        cv::Scalar color(0, 255, 0);
        std::string cat = "None";
        if (det.classifications.size() > 0)
        {
            cat_conf = det.classifications[0].conf;
            if (det.classifications[0].cat)
            {
                color = det.classifications[0].cat->color;
                cat = det.classifications[0].cat->getName();
            }
        }
        addOrUpdate((*image)(det.bounding_box), det.id, ts, cat_conf, det.confidence, color, cat);
    }
}

int DetectionOverlay::getWidth() const
{
    if (max_num_tiles > 0)
    {
        return static_cast<int32_t>(image->getSize().width / max_num_tiles);
    }
    else
    {
        return 100;
    }
}

void DetectionOverlay::addOrUpdate(const aq::SyncedMemory& patch_,
                                   uint32_t id,
                                   const mo::Time_t& ts,
                                   float cat_conf,
                                   float det_conf,
                                   cv::Scalar color,
                                   const std::string& classification)
{
    const auto width = getWidth();
    aq::SyncedMemory patch;
    if (patch_.getSyncState() < patch_.DEVICE_UPDATED)
    {
        cv::Mat resized;
        cv::resize(patch_.getMat(stream()), resized, cv::Size(width, width), 0.0, 0.0, cv::INTER_LINEAR);
        patch = aq::SyncedMemory(resized);
    }
    else
    {
        cv::cuda::GpuMat resized;
        cv::cuda::resize(
            patch_.getGpuMat(stream()), resized, cv::Size(width, width), 0.0, 0.0, cv::INTER_NEAREST, stream());
        stream().waitForCompletion();
        patch = aq::SyncedMemory(resized);
    }

    auto itr = m_detection_patches.find(id);
    if (itr == m_detection_patches.end())
    {
        m_detection_patches[id] = {patch, ts, ts, cat_conf, det_conf, color, classification};
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

template <>
void DetectionOverlay::drawOverlay(mo::Context* ctx)
{
    cv::Mat im;
    image->clone(im);

    const uint32_t width = static_cast<uint32_t>(im.cols) / max_num_tiles;
    for (size_t i = 0; i < m_draw_locations.size(); ++i)
    {
        if ((i * width + width) < im.cols)
        {
            const auto patch_itr = m_detection_patches.find(m_draw_locations[i]);
            if (patch_itr != m_detection_patches.end())
            {
                const auto roi = patch_itr->second.patch.getMat(ctx);
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
    output_param.updateData(im, mo::tag::_param = image_param);
}

void DetectionOverlay::pruneOldDetections(const mo::Time_t& ts)
{
    // const auto size = image->getSize();
    for (auto itr = m_draw_locations.begin(); itr != m_draw_locations.end();)
    {
        auto patch_itr = m_detection_patches.find(*itr);
        if (patch_itr != m_detection_patches.end())
        {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(ts - patch_itr->second.last_seen_time).count() >
                (max_age_seconds * 1000.0F))
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
}
}

using namespace aq::nodes;
MO_REGISTER_CLASS(DetectionOverlay)
