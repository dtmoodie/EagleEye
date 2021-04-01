/*
#include "CorrelationTracker.hpp"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <dlib/opencv.h>
namespace aqdlib
{

    template <class DetType>
    void DlibCorrelationTracker::apply(const cv::Mat& img)
    {
        auto ts = detections_param.getNewestTimestamp();
        auto fn = detections_param.getNewestFrameNumber();
        const DetType* in = mo::get<const DetType*>(detections);
        dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
        if ((ts == image_param.getTimestamp() || fn == image_param.getFrameNumber()) && in)
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

    aq::Algorithm::InputState DlibCorrelationTracker::checkInputs()
    {

        if (detections_param.getData())
        {
            boost::optional<mo::Header> header = detections_param.getNewestHeader();
            if (header)
            {
                image_param.getData(header.get_ptr());
                return aq::Algorithm::InputState::kALL_VALID;
            }
            else
            {
                return aq::Algorithm::InputState::kNONE_VALID;
            }
        }

        if (image_param.getData())
        {
            return aq::Algorithm::InputState::kALL_VALID;
        }

        return aq::Algorithm::InputState::kNOT_UPDATED;
    }

    bool DlibCorrelationTracker::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        const cv::Mat img = this->image->getMat(stream.get());
        dlib::cv_image<dlib::bgr_pixel> dlib_img(img);

        boost::optional<mo::Header> image_header = image_param.getNewestHeader();
        boost::optional<mo::Header> detection_header = detections_param.getNewestHeader();

        if (!image_header)
        {
            return false;
        }

        if (!detection_header)
        {
            // no detections at all?
            // if we have previous detections, do the track
        }

        if (*image_header == *detection_header)
        {
            const uint32_t num_entities = this->detections->getNumEntities();
            m_tracked_objects = *this->detections;

            auto bbs = m_tracked_objects.getComponent<aq::detection::BoundingBox2d>();
            mt::Tensor<dlib::correlation_tracker, 1> trackers =
                m_tracked_objects.getComponentMutable<dlib::correlation_tracker>();

            for (uint32_t i = 0; i < num_entities; ++i)
            {
                cv::Rect2f bb = bbs[i];
                boundingBoxToPixels(bb, image->size());
                dlib::rectangle rect(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
                trackers[i].start_track(dlib_img, rect);
            }

            return true;
        }

        if (*image_header > *detection_header)
        {
            // image is newer, track what we've already detected
        }

        return true;
    }

} // namespace aqdlib

using namespace aqdlib;
MO_REGISTER_CLASS(DlibCorrelationTracker)
*/