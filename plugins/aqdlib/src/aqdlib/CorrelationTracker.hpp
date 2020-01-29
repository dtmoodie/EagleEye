#pragma once

#include <Aquila/types/SyncedMemory.hpp>
#include <aqcore/detection/DetectionFilter.hpp>
#include <aqcore/detection/DetectionTracker.hpp>
#include <dlib/image_processing/correlation_tracker.h>

namespace aq
{
namespace nodes
{

class DlibCorrelationTracker : public DetectionTracker
{
  public:
    MO_DERIVE(DlibCorrelationTracker, DetectionTracker)
        INPUT(aq::SyncedMemory, image, nullptr)
        MULTI_INPUT(detections, aq::DetectedObjectSet, aq::DetectionDescriptionSet)
        APPEND_FLAGS(detections, mo::ParamFlags::Optional_e)

        MULTI_OUTPUT(output, aq::DetectedObjectSet, aq::DetectionDescriptionSet)
    MO_END

    template <class DetType>
    void apply(const cv::Mat&);

    template <class CTX>
    bool processImpl(CTX* ctx);

  protected:
    virtual InputState checkInputs() override;

    virtual bool processImpl() override;

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
};
}
}
