#include "Frame.h"
#include "Aquila/rcc/external_includes/cv_cudawarping.hpp"
#include "Aquila/rcc/external_includes/cv_cudaarithm.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace aq;
using namespace aq::nodes;


FrameRate::FrameRate():
    m_framerate_rolling_mean(boost::accumulators::tag::rolling_window::window_size = 30){
    
}

bool FrameRate::processImpl(){
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    prevTime = currentTime;
    double inst_frame_rate = 1000.0 / static_cast<double>(delta.total_milliseconds());
    m_framerate_rolling_mean(inst_frame_rate);
    double mean_fr = boost::accumulators::rolling_mean(m_framerate_rolling_mean);
    framerate_param.updateData(mean_fr);
    auto ts = input_param.getTimestamp();
    if(ts && _previous_frame_timestamp)
    {
        auto ts_delta = std::chrono::duration_cast<std::chrono::milliseconds>(*ts - *_previous_frame_timestamp);
        frametime = ts_delta;
        frametime_param.emitUpdate(input_param);
    }
    _previous_frame_timestamp = input_param.getTimestamp();
    if(draw_fps && output_param.hasSubscriptions()){
        cv::Mat draw_roi(25, 100, input->getType());
        draw_roi.setTo(cv::Scalar::all(0));
        std::stringstream ss;
        ss << std::setprecision(4) << mean_fr;
        cv::putText(draw_roi, ss.str(), cv::Point(0, 23), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
        SyncedMemory::SYNC_STATE state = input->getSyncState();
        if(state == SyncedMemory::HOST_UPDATED){
            cv::Mat draw_img;
            input->clone(draw_img, stream());
            draw_roi.copyTo(draw_img(cv::Rect(cv::Point(0,0), cv::Size(100, 25))));
            output_param.updateData(draw_img, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
        }else{
            cv::cuda::GpuMat draw_gpu;
            draw_gpu.upload(draw_roi, stream());
            cv::cuda::GpuMat draw_img;
            input->clone(draw_img, stream());
            draw_gpu.copyTo(draw_img(cv::Rect(cv::Point(0,0), cv::Size(100, 25))), stream());
            output_param.updateData(draw_img, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
        }
    }
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
        if(_prev_time)
        {
            if(*cur_time < *_prev_time)
            {
                MO_LOG(warning) << "Received frame that is " << *_prev_time - *cur_time << " older than last frame";
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
