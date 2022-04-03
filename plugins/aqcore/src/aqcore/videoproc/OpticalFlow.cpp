#include "OpticalFlow.h"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/core/metaobject_config.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace aq
{
    namespace nodes
    {

        std::vector<cv::cuda::GpuMat> IPyrOpticalFlow::makePyramid(const cv::cuda::GpuMat& mat,
                                                                   cv::cuda::Stream& stream) const
        {
            std::vector<cv::cuda::GpuMat> pyramid;
            cv::cuda::GpuMat gray;
            const cv::Size window_size_(window_size, window_size);
            if (mat.channels() != 1)
            {
                cv::cuda::cvtColor(mat, gray, cv::COLOR_BGR2GRAY, 0, stream);
            }
            else
            {
                gray = mat;
            }
            // TODO
            return pyramid;
        }

        std::vector<cv::Mat> IPyrOpticalFlow::makePyramid(const cv::Mat& mat) const
        {
            cv::Mat gray;
            std::vector<cv::Mat> pyramid;
            const cv::Size window_size_(window_size, window_size);
            if (mat.channels() != 1)
            {
                cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray = mat;
            }
            cv::buildOpticalFlowPyramid(gray, pyramid, window_size_, pyramid_levels);
            return pyramid;
        }

        bool DetectionLandmarkTracker::processImpl(mo::IAsyncStream& stream)
        {
            cv::Mat in = image->getMat(&stream);
            std::vector<cv::Mat> pyramid = this->makePyramid(in);

            const boost::optional<mo::Header> current_header = image_param.getNewestHeader();
            MO_ASSERT(current_header && "Unable to operate on data with no header information");
            MO_ASSERT(detections_param.checkFlags(mo::ParamFlags::kREQUIRE_BUFFERED));
            if (m_prev_cpu_pyramid.empty())
            {
                m_previous_header = current_header;
                m_prev_cpu_pyramid = std::move(pyramid);
                aq::TDetectedObjectSet<Components_t> current_landmarks = *detections;
                this->output.publish(std::move(current_landmarks), mo::tags::header = *current_header);
                return true;
            }
            else
            {

                if (current_header && m_previous_header)
                {
                    auto previous_landmark_ecs = detections_param.getTypedData(m_previous_header.get_ptr(), &stream);

                    if (previous_landmark_ecs)
                    {
                        const cv::Size window_size_(window_size, window_size);

                        mt::Tensor<const cv::Point2f, 2> previous_landmarks =
                            previous_landmark_ecs->data.getComponent<aq::detection::LandmarkDetection>();
                        const uint32_t num_entities = previous_landmarks.getShape()[0];
                        const uint32_t num_points = num_entities * previous_landmarks.getShape()[1];

                        cv::Mat_<cv::Point2f> wrapped(
                            num_points, 1, const_cast<cv::Point2f*>(previous_landmarks.data()));

                        auto output = *detections;

                        cv::Mat status, error;

                        mt::Tensor<cv::Point2f, 2> tracked_landmarks =
                            output.getComponentMutable<aq::detection::LandmarkDetection>();

                        cv::Mat_<cv::Point2f> tracked_points(num_points, 1, tracked_landmarks.data());

                        cv::calcOpticalFlowPyrLK(m_prev_cpu_pyramid,
                                                 pyramid,
                                                 wrapped,
                                                 tracked_points,
                                                 status,
                                                 error,
                                                 window_size_,
                                                 pyramid_levels);

                        m_previous_header = current_header;
                        m_prev_cpu_pyramid = std::move(pyramid);
                        this->output.publish(std::move(output), mo::tags::header = *current_header);
                        return true;
                    }
                }
            }
            return false;
        }

        bool DetectionLandmarkTracker::processImpl(mo::IDeviceStream& stream)
        {
            const aq::SyncedMemory::SyncState state = image->state();
            if (state < aq::SyncedMemory::SyncState::DEVICE_UPDATED)
            {
                return processImpl(static_cast<mo::IAsyncStream&>(stream));
            }
            // TODO
            return false;
        }

        bool DetectionLandmarkTracker::processImpl()
        {
            // asdf
            return true;
        }

    } // namespace nodes
} // namespace aq

using namespace aq::nodes;

MO_REGISTER_CLASS(DetectionLandmarkTracker)
