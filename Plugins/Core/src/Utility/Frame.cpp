#include "Frame.h"
#include "Aquila/rcc/external_includes/cv_cudawarping.hpp"
#include "Aquila/rcc/external_includes/cv_cudaarithm.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"

using namespace aq;
using namespace aq::nodes;


bool FrameRate::processImpl()
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    prevTime = currentTime;
    framerate_param.updateData(1000.0 / (double)delta.total_milliseconds());
    auto ts = input_param.getTimestamp();
    if(ts && _previous_frame_timestamp)
    {
        mo::Time_t ts_delta =  *ts - *_previous_frame_timestamp;
        frametime = ts_delta;
        frametime_param.emitUpdate(input_param);
    }
    _previous_frame_timestamp = input_param.getTimestamp();
    return true;
}
MO_REGISTER_CLASS(FrameRate)

bool DetectFrameSkip::processImpl()
{
    auto cur_time = input_param.getTimestamp();

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

bool FrameLimiter::processImpl()
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

bool CreateMat::processImpl()
{
    if(data_type_param.modified() || channels_param.modified() || width_param.modified() || height_param.modified() || fill_param.modified())
    {
        cv::cuda::GpuMat mat;
        mat.create(height, width, CV_MAKETYPE(data_type.getValue(), channels));
        output_param.updateData(mat);
        data_type_param.modified(false);
        channels_param.modified(false);
        width_param.modified(false);
        height_param.modified(false);
        fill_param.modified(false);
    }
    output.getGpuMatMutable(stream()).setTo(fill, stream());
    return true;
}
MO_REGISTER_CLASS(CreateMat)

bool SetMatrixValues::processImpl()
{

    /*if(mask)
    {
        input->getGpuMatMutable(stream()).setTo(replace_value, mask->getGpuMat(stream()), stream());
    }else
    {
        input->getGpuMatMutable(stream()).setTo(replace_value, stream());
    }*/
    return true;
}
//MO_REGISTER_CLASS(SetMatrixValues)

bool Resize::processImpl()
{
    if(input && !input->empty())
    {
        if (input->getSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            cv::Mat resized;
            cv::resize(input->getMat(stream()), resized, cv::Size(width, height), 0.0, 0.0, interpolation_method.getValue());
            output_param.updateData(resized, input_param.getTimestamp(), _ctx.get());
            return true;
        }
        else
        {
            cv::cuda::GpuMat resized;
            cv::cuda::resize(input->getGpuMat(stream()), resized, cv::Size(width, height), 0.0, 0.0, interpolation_method.getValue(), stream());
            output_param.updateData(resized, input_param.getTimestamp(), _ctx.get());
            return true;
        }
    }
    return false;
}
MO_REGISTER_CLASS(Resize)

bool Subtract::processImpl()
{
    if (input->getSyncState() < SyncedMemory::DEVICE_UPDATED)
    {
        cv::subtract(input->getMat(stream()), value, output.getMatMutable(stream()), mask ? mask->getMat(stream()) : cv::noArray(),  dtype.getValue());
    }else
    {
        cv::cuda::subtract(input->getGpuMat(stream()), value, output.getGpuMatMutable(stream()), mask ? mask->getGpuMat(stream()) : cv::noArray(), dtype.getValue(), stream());
    }
    return true;
}
MO_REGISTER_CLASS(Subtract)

bool RescaleContours::processImpl()
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
    output_param.emitUpdate(input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(RescaleContours)
