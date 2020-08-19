#include "OpticalFlow.h"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/core/metaobject_config.hpp>

namespace aq
{
    namespace nodes
    {

        template <>
        bool PyrLKLandmarkTracker::processImpl(mo::IAsyncStream& stream)
        {
            cv::Mat in = input->getMat(&stream);
            cv::Mat gray;
            if (in.channels() != 1)
            {
                cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray = in;
            }
            MO_ASSERT(detections_param.checkFlags(mo::ParamFlags::kREQUIRE_BUFFERED));
            if (m_prev_pyramid.empty())
            {
                m_prev_pyramid = {SyncedImage(gray)};
                m_prev_time = input_param.getNewestTimestamp();
                return true;
            }
            else
            {
                const boost::optional<mo::Header> header_ = input_param.getNewestHeader();
                if (header_)
                {
                    const mo::Header& hdr = *header_;
                    const mo::Header desired_header(hdr.frame_number - 1);
                    mo::TDataContainerConstPtr_t<aq::EntityComponentSystem> previous_landmark_ecs =
                        detections_param.getTypedData(&desired_header, &stream);
                    if (previous_landmark_ecs)
                    {
                        mt::Tensor<const cv::Point2f, 2> previous_landmarks =
                            previous_landmark_ecs->data.getComponent<aq::detection::LandmarkDetection>();
                        const uint32_t num_entities = previous_landmarks.getShape()[0];
                        const uint32_t num_points = num_entities * previous_landmarks.getShape()[1];

                        cv::Mat_<cv::Point2f> wrapped(
                            num_points, 1, const_cast<cv::Point2f*>(previous_landmarks.data()));

                        aq::TEntityComponentSystem<Components_t> output = *detections;

                        std::vector<cv::Mat> pyramid;
                        for (auto& synced_image : m_prev_pyramid)
                        {
                            pyramid.push_back(synced_image.getMat(&stream));
                        }

                        cv::Mat status, error;
                        cv::Size size(window_size, window_size);
                        mt::Tensor<cv::Point2f, 2> tracked_landmarks =
                            output.getComponentMutable<aq::detection::LandmarkDetection>();
                        cv::Mat_<cv::Point2f> tracked_points(num_points, 1, tracked_landmarks.data());

                        cv::calcOpticalFlowPyrLK(
                            pyramid, gray, wrapped, tracked_points, status, error, size, pyramid_levels);

                        this->output.publish(std::move(output), mo::tags::param = &input_param);
                        return true;
                    }
                }
            }
            return false;
        }

        bool PyrLKLandmarkTracker::processImpl()
        {
            std::shared_ptr<mo::IAsyncStream> stream = this->getStream();
            return nodeStreamSwitch(this, *stream);
        }

        template <>
        bool PyrLKLandmarkTracker::processImpl(mo::IDeviceStream& stream)
        {
            const aq::SyncedMemory::SyncState state = input->state();
            if (state < aq::SyncedMemory::SyncState::DEVICE_UPDATED)
            {
                return processImpl(static_cast<mo::IAsyncStream&>(stream));
            }
            // TODO
            return false;
        }

    } // namespace nodes
} // namespace aq

using namespace aq::nodes;

MO_REGISTER_CLASS(PyrLKLandmarkTracker)
