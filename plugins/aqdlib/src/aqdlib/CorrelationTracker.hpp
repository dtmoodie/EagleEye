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

      protected:
        aq::Algorithm::InputState checkInputs() override;

        bool processImpl() override;

        using TrackState_t = ct::VariadicTypedef<aq::detection::BoundingBox2d, dlib::correlation_tracker>;
        aq::TDetectedObjectSet<TrackState_t> m_tracked_objects;
    };

} // namespace aqdlib

namespace ct
{
    REFLECT_BEGIN(dlib::correlation_tracker)
    PROPERTY(get_filter_size)
    PROPERTY(get_num_scale_levels)
    PROPERTY(get_scale_window_size)
    PROPERTY(get_regularizer_space)
    PROPERTY(get_regularizer_scale)
    PROPERTY(get_scale_pyramid_alpha)
    REFLECT_END;
} // namespace ct

#endif // AQDLIB_CORRELATION_TRACKER_HPP
