#pragma once
#include "../OpenCVCudaNode.hpp"

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/types/DetectionDescription.hpp>

#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/params/TMultiSubscriber.hpp>

#include <boost/circular_buffer.hpp>

namespace aqcore
{

    class DetectionOverlay : public OpenCVCudaNode
    {
      public:
        MO_DERIVE(DetectionOverlay, OpenCVCudaNode)
            INPUT(aq::SyncedImage, image)
            INPUT(aq::DetectedObjectSet, detections)

            PARAM(float, max_age_seconds, 5.0F)
            PARAM(uint32_t, max_num_tiles, 10)

            PARAM(bool, draw_conf, true)
            PARAM(bool, draw_classification, true)
            PARAM(bool, draw_timestamp, true)
            PARAM(bool, draw_age, true)

            OUTPUT(aq::SyncedImage, output, {})
        MO_END;

        template <class DetType, class CTX>
        void apply(CTX* ctx);

        template <class CTX>
        bool processImpl(CTX& ctx);

      protected:
        virtual bool processImpl() override;

        void updateOverlay(const aq::DetectedObjectSet& dets, const mo::Time& ts, mo::IAsyncStream&);

        void drawOverlay(mo::IAsyncStream& ctx);
        void drawOverlay(mo::IDeviceStream& ctx);

        void addOrUpdate(const aq::SyncedImage& patch,
                         aq::detection::Id id,
                         const mo::Time& ts,
                         float cat_conf,
                         aq::detection::Confidence det_conf,
                         cv::Scalar color,
                         const std::string& classification,
                         const aq::SyncedImage& patch_source,
                         mo::IAsyncStream&);

      private:
        void pruneOldDetections(const mo::Time& ts);
        struct RenderedDet
        {
            aq::SyncedImage patch;
            mo::Time first_seen_time;
            mo::Time last_seen_time;
            float cat_conf;
            float det_conf;
            cv::Scalar color;
            std::string classification;
            // in case patch is not an owning view, this is an owning view to the data referred to by patch
            aq::SyncedImage patch_source;
        };

        int getWidth() const;
        std::vector<uint32_t> m_draw_locations;
        std::unordered_map<uint32_t, RenderedDet> m_detection_patches;
    };

} // namespace aqcore
