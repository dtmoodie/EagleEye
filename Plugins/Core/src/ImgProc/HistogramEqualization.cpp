#include "HistogramEqualization.hpp"
#include <Aquila/Nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>

using namespace aq::Nodes;

bool HistogramEqualization::ProcessImpl()
{
    cv::cuda::GpuMat output;
    std::vector<cv::cuda::GpuMat> channels;
    if(!per_channel)
    {
        cv::cuda::GpuMat hsv;
        cv::cuda::cvtColor(input->GetGpuMat(Stream()), hsv, cv::COLOR_BGR2HSV, 0, Stream());
        cv::cuda::split(hsv, channels, Stream());
        cv::cuda::GpuMat equalized;
        cv::cuda::equalizeHist(channels[2], equalized, Stream());
        channels[2] = equalized;
        cv::cuda::merge(channels, hsv, Stream());
        cv::cuda::cvtColor(hsv, output, cv::COLOR_HSV2BGR, 0, Stream());
    }else
    {
        cv::cuda::split(input->GetGpuMat(Stream()), channels, Stream());
        for(int i = 0; i < channels.size(); ++i)
        {
            cv::cuda::GpuMat equalized;
            cv::cuda::equalizeHist(channels[i], equalized, Stream());
            channels[i] = equalized;
        }
        cv::cuda::merge(channels, output, Stream());
    }

    output_param.UpdateData(output, input_param.GetTimestamp(), _ctx);
    return true;
}

MO_REGISTER_CLASS(HistogramEqualization)

bool CLAHE::ProcessImpl()
{
    if(!_clahe || clip_limit_param._modified || grid_size_param._modified)
    {
        _clahe = cv::cuda::createCLAHE(clip_limit, cv::Size(grid_size, grid_size));
        clip_limit_param._modified = false;
        grid_size_param._modified = false;
    }
    cv::cuda::GpuMat hsv;
    cv::cuda::cvtColor(input->GetGpuMat(Stream()), hsv, cv::COLOR_BGR2HSV, 0, Stream());
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(hsv, channels, Stream());
    _clahe->apply(channels[2], channels[2], Stream());
    cv::cuda::GpuMat output;
    cv::cuda::merge(channels, hsv, Stream());
    cv::cuda::cvtColor(hsv, output, cv::COLOR_HSV2BGR, 0, Stream());
    output_param.UpdateData(output, input_param.GetTimestamp(), _ctx);
    return true;
}

MO_REGISTER_CLASS(CLAHE)
