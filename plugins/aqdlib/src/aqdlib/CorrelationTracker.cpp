#include "CorrelationTracker.hpp"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <dlib/opencv.h>
namespace aq
{
namespace nodes
{

void DlibCorrelationTracker::TrackState::readMetadata(const aq::DetectedObject& det)
{
    classifications = det.classifications;
    detection_id = det.id;
    detection_confidence = det.confidence;
}

void DlibCorrelationTracker::TrackState::readMetadata(const aq::DetectionDescription& det)
{
    track_description = det.descriptor;
    readMetadata(static_cast<const aq::DetectedObject&>(det));
}

void DlibCorrelationTracker::TrackState::writeMetadata(aq::DetectedObject& det)
{
    det.id = detection_id;
    det.classifications = classifications;
    det.confidence = detection_confidence;
}

void DlibCorrelationTracker::TrackState::writeMetadata(aq::DetectionDescription& det)
{
    det.descriptor = track_description;
    writeMetadata(static_cast<aq::DetectedObject&>(det));
}

template <class DetType>
void DlibCorrelationTracker::apply(const cv::Mat& img)
{
    auto ts = detections_param.getInputTimestamp();
    auto fn = detections_param.getInputFrameNumber();
    const DetType* in = mo::get<const DetType*>(detections);
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
    if ((ts == image_param.getTimestamp() || fn == image_param.getFrameNumber()) && in )
    {
        // Initialize new tracks
        const auto num_dets = in->size();
        m_trackers.clear();
        m_trackers.resize(num_dets);
        for (size_t i = 0; i < num_dets; ++i)
        {
            cv::Rect2f bb = (*in)[i].bounding_box;
            boundingBoxToPixels(bb, img.size());
            dlib::rectangle rect(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
            m_trackers[i].tracker.start_track(dlib_img, rect);
        }
        output_param.updateData(*in, mo::tag::_param = image_param);
    }
    else
    {
        // track
        DetType output;
        for (size_t i = 0; i < m_trackers.size(); ++i)
        {
            m_trackers[i].tracker.update(dlib_img);
            ++m_trackers[i].track_count;
            dlib::drectangle rect = m_trackers[i].tracker.get_position();
            typename DetType::value_type tracked_det;
            tracked_det.bounding_box.x = rect.left();
            tracked_det.bounding_box.y = rect.top();
            tracked_det.bounding_box.width = rect.width();
            tracked_det.bounding_box.height = rect.height();
            m_trackers[i].writeMetadata(tracked_det);
            output.push_back(tracked_det);
        }
        output_param.updateData(std::move(output), mo::tag::_param = image_param);
    }
}

Algorithm::InputState DlibCorrelationTracker::checkInputs()
{
    if(detections_param.modified())
    {
        if(detections_param.getInput(mo::OptionalTime_t()))
        {
            auto ts = detections_param.getTimestamp();
            if(image_param.getInput(ts))
            {
                return Algorithm::InputState::AllValid;
            }else
            {
                return Algorithm::InputState::NoneValid;
            }
        }
    }
    if(image_param.modified())
    {
        if(image_param.getInput(mo::OptionalTime_t()))
        {
            return Algorithm::InputState::AllValid;
        }
    }
    return Algorithm::InputState::NotUpdated;
}

template <>
bool DlibCorrelationTracker::processImpl(mo::Context* ctx)
{
    cv::Mat img = image->getMat(ctx, 0);
    mo::selectType<decltype(detections_param)::TypeTuple>(*this, detections_param.getTypeInfo(), img);
    return true;
}

template <>
bool DlibCorrelationTracker::processImpl(mo::CvContext* ctx)
{
    bool sync = false;
    cv::Mat img = image->getMat(ctx, 0, &sync);
    if (sync)
    {
        ctx->getStream().waitForCompletion();
    }
    mo::selectType<decltype(detections_param)::TypeTuple>(*this, detections_param.getTypeInfo(), img);
    return true;
}

bool DlibCorrelationTracker::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}
}
}

using namespace aq::nodes;
MO_REGISTER_CLASS(DlibCorrelationTracker)
