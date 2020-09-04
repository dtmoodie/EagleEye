#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/rcc/external_includes/cv_core.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <MetaObject/params/ParamMacros.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class FrameRate : public Node
        {
          public:
            FrameRate();
            MO_DERIVE(FrameRate, Node)
                STATUS(double, framerate, 0.0)
                STATUS(std::chrono::milliseconds, frametime, {})
                PARAM(bool, draw_fps, true)
                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, output, {})
            MO_END;

          protected:
            virtual bool processImpl() override;
            boost::posix_time::ptime prevTime;
            boost::optional<mo::Time_t> _previous_frame_timestamp;
            boost::accumulators::accumulator_set<double,
                                                 boost::accumulators::stats<boost::accumulators::tag::rolling_mean>>
                m_framerate_rolling_mean;
        };

        class DetectFrameSkip : public Node
        {
          public:
            MO_DERIVE(DetectFrameSkip, Node)
                INPUT(SyncedImage, input)
            MO_END;

          protected:
            virtual bool processImpl() override;
            boost::optional<mo::Time_t> _prev_time;
            boost::optional<mo::Time_t> _initial_time; // used to zero base time
        };

        class FrameLimiter : public Node
        {
          public:
            MO_DERIVE(FrameLimiter, Node)
                INPUT(SyncedImage, input)

                PARAM(double, desired_framerate, 30.0)

                OUTPUT(SyncedImage, output, {})
            MO_END;

          protected:
            virtual bool processImpl() override;
            boost::posix_time::ptime last_time;
        };

        class CreateMat : public Node
        {
          public:
            MO_DERIVE(CreateMat, Node)
                ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_32S, CV_32F, CV_64F)
                PARAM(int, channels, 1)
                PARAM(int, width, 1920)
                PARAM(int, height, 1080)
                PARAM(cv::Scalar, fill, cv::Scalar::all(0))
                OUTPUT(SyncedImage, output)
            MO_END;

          protected:
            virtual bool processImpl() override;
        };

        class SetMatrixValues : public Node
        {
          public:
            MO_DERIVE(SetMatrixValues, Node)
                INPUT(SyncedImage, input)
                OPTIONAL_INPUT(SyncedImage, mask)
                PARAM(cv::Scalar, replace_value, cv::Scalar::all(0))
            MO_END;

          protected:
            virtual bool processImpl() override;
        };

        class Resize : public Node
        {
          public:
            MO_DERIVE(Resize, Node)
                INPUT(SyncedImage, input)
                ENUM_PARAM(interpolation_method,
                           cv::INTER_NEAREST,
                           cv::INTER_LINEAR,
                           cv::INTER_CUBIC,
                           cv::INTER_AREA,
                           cv::INTER_LANCZOS4,
                           cv::INTER_MAX)
                PARAM(int, width, 224)
                PARAM(int, height, 224)
                OUTPUT(SyncedImage, output)
            MO_END;

          protected:
            virtual bool processImpl() override;
        };

        class RescaleContours : public Node
        {
          public:
            MO_DERIVE(RescaleContours, Node)
                INPUT(std::vector<std::vector<cv::Point>>, input)
                OUTPUT(std::vector<std::vector<cv::Point>>, output)
                PARAM(float, scale_x, 1.0)
                PARAM(float, scale_y, 1.0)
            MO_END;

          protected:
            virtual bool processImpl() override;
        };

        class Subtract : public Node
        {
          public:
            MO_DERIVE(Subtract, Node)

                INPUT(SyncedImage, input)
                OPTIONAL_INPUT(SyncedMemory, mask)

                PARAM(cv::Scalar, value, cv::Scalar::all(0))
                ENUM_PARAM(dtype, CV_8U, CV_16S, CV_16U, CV_32S, CV_32F, CV_64F)

                OUTPUT(SyncedImage, output)
            MO_END;

          protected:
            virtual bool processImpl() override;
        };

    } // namespace nodes
} // namespace aq