#pragma once
#include <src/precompiled.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq{
namespace nodes{
    class FrameRate: public Node{
    public:
        FrameRate();
        MO_DERIVE(FrameRate, Node)
            STATUS(double, framerate, 0.0)
            STATUS(std::chrono::milliseconds, frametime, {})
            PARAM(bool, draw_fps, true)
            INPUT(SyncedMemory, input, nullptr)
            OUTPUT(SyncedMemory, output, {})
        MO_END
    protected:
        bool processImpl();
        boost::posix_time::ptime prevTime;
        boost::optional<mo::Time_t> _previous_frame_timestamp;
        boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::rolling_mean>>  m_framerate_rolling_mean;
    };

    class DetectFrameSkip: public Node{
    public:
        MO_DERIVE(DetectFrameSkip, Node)
            INPUT(SyncedMemory, input, nullptr)
        MO_END;
    protected:
        bool processImpl();
        boost::optional<mo::Time_t> _prev_time;
        boost::optional<mo::Time_t> _initial_time; // used to zero base time
    };

    class FrameLimiter : public Node{
    public:
        MO_DERIVE(FrameLimiter, Node)
            PARAM(double, desired_framerate, 30.0)
        MO_END
    protected:
        bool processImpl();
        boost::posix_time::ptime lastTime;
    };

    class CreateMat: public Node{
    public:
        MO_DERIVE(CreateMat, Node)
            ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_32S, CV_32F, CV_64F)
            PARAM(int, channels, 1)
            PARAM(int, width, 1920)
            PARAM(int, height, 1080)
            PARAM(cv::Scalar, fill, cv::Scalar::all(0))
            OUTPUT(SyncedMemory, output, SyncedMemory())
        MO_END;
    protected:
        bool processImpl();

    };

    class SetMatrixValues: public Node{
    public:
        MO_DERIVE(SetMatrixValues, Node)
            INPUT(SyncedMemory, input, nullptr)
            OPTIONAL_INPUT(SyncedMemory, mask, nullptr)
            PARAM(cv::Scalar, replace_value, cv::Scalar::all(0))
    protected:
        bool processImpl();
    };

    class Resize : public Node{
    public:
        MO_DERIVE(Resize, Node)
            INPUT(SyncedMemory, input, nullptr)
            ENUM_PARAM(interpolation_method, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_AREA, cv::INTER_LANCZOS4, cv::INTER_MAX)
            PARAM(int, width, 224)
            PARAM(int, height, 224)
            OUTPUT(SyncedMemory, output, SyncedMemory())
        MO_END
    protected:
        bool processImpl();
    };

    class RescaleContours: public Node
    {
    public:
        MO_DERIVE(RescaleContours, Node)
            INPUT(std::vector<std::vector<cv::Point>>, input, nullptr)
            OUTPUT(std::vector<std::vector<cv::Point>>, output, {})
            PARAM(float, scale_x, 1.0)
            PARAM(float, scale_y, 1.0)
        MO_END
    protected:
        bool processImpl();
    };

    class Subtract : public Node
    {
    public:
        MO_DERIVE(Subtract, Node)
            PARAM(cv::Scalar, value, cv::Scalar::all(0))
            INPUT(SyncedMemory, input, nullptr)
            ENUM_PARAM(dtype, CV_8U, CV_16S, CV_16U, CV_32S, CV_32F, CV_64F)
            OPTIONAL_INPUT(SyncedMemory, mask, nullptr)
            OUTPUT(SyncedMemory, output, SyncedMemory())
        MO_END
    protected:
        bool processImpl();
    };
}
}
