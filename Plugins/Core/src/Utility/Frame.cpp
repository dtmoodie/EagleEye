#include "Frame.h"
#include "Aquila/rcc/external_includes/cv_cudawarping.hpp"
#include "Aquila/rcc/external_includes/cv_cudaarithm.hpp"
#include <Aquila/Qualifiers.hpp>
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"

using namespace aq;
using namespace aq::Nodes;


bool FrameRate::ProcessImpl()
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    prevTime = currentTime;
    framerate_param.UpdateData(1000.0 / (double)delta.total_milliseconds());
    auto ts = input_param.GetTimestamp();
    if(ts && _previous_frame_timestamp)
    {
        mo::time_t ts_delta =  *ts - *_previous_frame_timestamp;
        frametime_param.UpdateData(ts_delta);
    }
    _previous_frame_timestamp = input_param.GetTimestamp();
    return true;
}
MO_REGISTER_CLASS(FrameRate)

bool DetectFrameSkip::ProcessImpl()
{
    auto cur_time = input_param.GetTimestamp();

    if(cur_time)
    {
        if(!_initial_time)
            _initial_time = cur_time;
        std::cout << *cur_time - *_initial_time << std::endl;
        if(_prev_time)
        {
            if(*cur_time < *_prev_time)
            {
                LOG(warning) << "Received frame that is " << *_prev_time - *cur_time << " older than last frame";
                return false;
            }
        }
        _prev_time = cur_time;
    }

    return true;
}
MO_REGISTER_CLASS(DetectFrameSkip)

bool FrameLimiter::ProcessImpl()
{
    auto currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta(currentTime - lastTime);
    lastTime = currentTime;
    int goalTime = 1000.0 / desired_framerate;
    if (delta.total_milliseconds() < goalTime)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(goalTime - delta.total_milliseconds()));
    }
    return true;
}
MO_REGISTER_CLASS(FrameLimiter)

bool CreateMat::ProcessImpl()
{
    if(data_type_param._modified || channels_param._modified || width_param._modified || height_param._modified || fill_param._modified)
    {
        cv::cuda::GpuMat mat;
        mat.create(height, width, CV_MAKETYPE(data_type.getValue(), channels));
        output_param.UpdateData(mat);
        data_type_param._modified = false;
        channels_param._modified = false;
        width_param._modified = false;
        height_param._modified = false;
        fill_param._modified = false;
    }
    output.getGpuMatMutable(Stream()).setTo(fill, Stream());
    return true;
}
MO_REGISTER_CLASS(CreateMat)

bool SetMatrixValues::ProcessImpl()
{

    /*if(mask)
    {
        input->getGpuMatMutable(Stream()).setTo(replace_value, mask->getGpuMat(Stream()), Stream());
    }else
    {
        input->getGpuMatMutable(Stream()).setTo(replace_value, Stream());
    }*/
    return true;
}
//MO_REGISTER_CLASS(SetMatrixValues)

bool Resize::ProcessImpl()
{
    if(input && !input->empty())
    {
        if (input->getSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            cv::Mat resized;
            cv::resize(input->getMat(Stream()), resized, cv::Size(width, height), 0.0, 0.0, interpolation_method.getValue());
            output_param.UpdateData(resized, input_param.GetTimestamp(), _ctx);
            return true;
        }
        else
        {
            cv::cuda::GpuMat resized;
            cv::cuda::resize(input->getGpuMat(Stream()), resized, cv::Size(width, height), 0.0, 0.0, interpolation_method.getValue(), Stream());
            output_param.UpdateData(resized, input_param.GetTimestamp(), _ctx);
            return true;
        }
    }
    return false;
}
MO_REGISTER_CLASS(Resize)

bool Subtract::ProcessImpl()
{
    if (input->getSyncState() < SyncedMemory::DEVICE_UPDATED)
    {
        cv::subtract(input->getMat(Stream()), value, output.getMatMutable(Stream()), mask ? mask->getMat(Stream()) : cv::noArray(),  dtype.getValue());
    }else
    {
        cv::cuda::subtract(input->getGpuMat(Stream()), value, output.getGpuMatMutable(Stream()), mask ? mask->getGpuMat(Stream()) : cv::noArray(), dtype.getValue(), Stream());
    }
    return true;
}
MO_REGISTER_CLASS(Subtract)

bool RescaleContours::ProcessImpl()
{
    output.resize(input->size());
    for(int i = 0; i < input->size(); ++i)
    {
        output[i].resize((*input)[i].size());
        for(int j = 0; j < (*input)[i].size(); ++j)
        {
            output[i][j].x = (*input)[i][j].x * scale_x;
            output[i][j].y = (*input)[i][j].y * scale_y;
        }
    }
    output_param.Commit(input_param.GetTimestamp(), _ctx);
    return true;
}

MO_REGISTER_CLASS(RescaleContours)
