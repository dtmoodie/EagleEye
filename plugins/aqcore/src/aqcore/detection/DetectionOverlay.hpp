#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/DetectionDescription.hpp>

#include <MetaObject/params/TMultiInput-inl.hpp>

#include <boost/circular_buffer.hpp>

namespace aq
{
namespace nodes
{
class DetectionOverlay : public Node
{
  public:
    MO_DERIVE(DetectionOverlay, Node)
        INPUT(aq::SyncedMemory, image, nullptr)
        MULTI_INPUT(detections, aq::DetectedObjectSet, aq::DetectionDescriptionSet, aq::DetectionDescriptionPatchSet)

        PARAM(float, max_age_seconds, 5.0F)
        PARAM(uint32_t, max_num_tiles, 10)

        PARAM(bool, draw_conf, true)
        PARAM(bool, draw_classification, true)
        PARAM(bool, draw_timestamp, true)
        PARAM(bool, draw_age, true)

        OUTPUT(aq::SyncedMemory, output, {})
    MO_END

    template <class DetType, class CTX>
    void apply(CTX* ctx);

    template <class CTX>
    bool processImpl(CTX* ctx);

  protected:
    virtual bool processImpl() override;

    void updateOverlay(const aq::DetectionDescriptionPatchSet& dets, const mo::Time_t& ts);
    template <class DetType>
    void updateOverlay(const DetType& dets, const mo::Time_t& ts);

    template <class CTX>
    void drawOverlay(CTX* ctx);

    void addOrUpdate(const aq::SyncedMemory& patch,
                     uint32_t id,
                     const mo::Time_t& ts,
                     float cat_conf,
                     float det_conf,
                     cv::Scalar color,
                     const std::string& classification);

  private:
    void pruneOldDetections(const mo::Time_t& ts);
    struct Det
    {
        aq::SyncedMemory patch;
        mo::Time_t first_seen_time;
        mo::Time_t last_seen_time;
        float cat_conf;
        float det_conf;
        cv::Scalar color;
        std::string classification;
    };

    int getWidth() const;
    std::vector<uint32_t> m_draw_locations;
    std::unordered_map<uint32_t, Det> m_detection_patches;
};
}
}
