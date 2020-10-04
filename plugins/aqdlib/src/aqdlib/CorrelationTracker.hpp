#ifndef AQDLIB_CORRELATION_TRACKER_HPP
#define AQDLIB_CORRELATION_TRACKER_HPP

#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <aqcore/detection/DetectionFilter.hpp>
#include <aqcore/detection/DetectionTracker.hpp>

#include <dlib/image_processing/correlation_tracker.h>

namespace aqdlib
{

    class DlibCorrelationTracker : public aqcore::DetectionTracker
    {
      public:
        using DesiredComponents_t = ct::VariadicTypedef<aq::detection::BoundingBox2d>;
        using Input_t = aq::TDetectedObjectSet<DesiredComponents_t>;

        MO_DERIVE(DlibCorrelationTracker, aqcore::DetectionTracker)
            INPUT(aq::SyncedImage, image)
            FLAGGED_INPUT(mo::ParamFlags::kOPTIONAL, Input_t, detections)

            OUTPUT(Input_t, output)
        MO_END;

        template <class DetType>
        void apply(const cv::Mat&);

        template <class CTX>
        bool processImpl(CTX* ctx);

      protected:
        InputState checkInputs() override;

        bool processImpl() override;

        struct TrackState
        {
            dlib::correlation_tracker tracker;
            size_t track_count = 0;

            void readMetadata(const aq::DetectedObject& det);
            void readMetadata(const aq::DetectionDescription& det);

            void writeMetadata(aq::DetectedObject& det);
            void writeMetadata(aq::DetectionDescription& det);

            // Metadata
            mo::SmallVec<Classification, 5> classifications;
            aq::SyncedMemory track_description;
            size_t detection_id;
            float detection_confidence;
        };

        std::vector<TrackState> m_trackers;
        Input_t m_trackers;
    };

} // namespace aqdlib

#endif // AQDLIB_CORRELATION_TRACKER_HPP