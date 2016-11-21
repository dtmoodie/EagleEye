#include "Frame.h"
#include "EagleLib/rcc/external_includes/cv_cudawarping.hpp"
#include "EagleLib/rcc/external_includes/cv_cudaarithm.hpp"
#include <EagleLib/Qualifiers.hpp>


using namespace EagleLib;
using namespace EagleLib::Nodes;


bool FrameRate::ProcessImpl()
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    prevTime = currentTime;
    framerate_param.UpdateData(1000.0 / delta.total_milliseconds());
    return true;
}

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

bool CreateMat::ProcessImpl()
{
    if(data_type_param.modified || channels_param.modified || width_param.modified || height_param.modified || fill_param.modified)
    {
        cv::cuda::GpuMat mat;
        mat.create(height, width, CV_MAKETYPE(data_type.getValue(), channels));
        output_param.UpdateData(mat);
        data_type_param.modified = false;
        channels_param.modified = false;
        width_param.modified = false;
        height_param.modified = false; 
        fill_param.modified = false;
    }
    output.GetGpuMatMutable(Stream()).setTo(fill, Stream());
    return true;
}

bool SetMatrixValues::ProcessImpl()
{
    if(mask)
    {
        input->GetGpuMatMutable(Stream()).setTo(replace_value, mask->GetGpuMat(Stream()), Stream());
    }else
    {
        input->GetGpuMatMutable(Stream()).setTo(replace_value, Stream());
    }
    return true;
}

bool Resize::ProcessImpl()
{
    if(input && !input->empty())
    {
        if (input->GetSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            cv::Mat resized;
            cv::resize(input->GetMat(Stream()), resized, cv::Size(width, height), 0.0, 0.0, interpolation_method.getValue());
            output_param.UpdateData(resized, input_param.GetTimestamp(), _ctx);
            return true;
        }
        else
        {
            cv::cuda::GpuMat resized;
            cv::cuda::resize(input->GetGpuMat(Stream()), resized, cv::Size(width, height), 0.0, 0.0, interpolation_method.getValue(), Stream());
            output_param.UpdateData(resized, input_param.GetTimestamp(), _ctx);
            return true;
        }
    }
    return false;
}

bool Subtract::ProcessImpl()
{
    if (input->GetSyncState() < SyncedMemory::DEVICE_UPDATED)
    {
        cv::subtract(input->GetMat(Stream()), value, output.GetMatMutable(Stream()), mask ? mask->GetMat(Stream()) : cv::noArray(),  dtype.getValue());
    }else
    {
        cv::cuda::subtract(input->GetGpuMat(Stream()), value, output.GetGpuMatMutable(Stream()), mask ? mask->GetGpuMat(Stream()) : cv::noArray(), dtype.getValue(), Stream());
    }
    return true;
}



MO_REGISTER_CLASS(SetMatrixValues)
MO_REGISTER_CLASS(FrameRate)
MO_REGISTER_CLASS(FrameLimiter)
MO_REGISTER_CLASS(CreateMat)
MO_REGISTER_CLASS(Resize)
MO_REGISTER_CLASS(Subtract)
