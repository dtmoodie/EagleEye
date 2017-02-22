#include "HistogramEqualization.hpp"
#include <EagleLib/Nodes/NodeInfo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
using namespace EagleLib::Nodes;

bool HistogramEqualization::ProcessImpl()
{
    cv::cuda::GpuMat hsv;
    cv::cuda::cvtColor(input->GetGpuMat(Stream()), hsv, cv::COLOR_BGR2HSV, 0, Stream());
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(hsv, channels, Stream());
    cv::cuda::equalizeHist(channels[2], channels[2], Stream());
    cv::cuda::GpuMat output;
    cv::cuda::merge(channels, hsv, Stream());
    cv::cuda::cvtColor(hsv, output, cv::COLOR_HSV2BGR, 0, Stream());
    output_param.UpdateData(output, input_param.GetTimestamp(), _ctx);
    return true;
}

MO_REGISTER_CLASS(HistogramEqualization)

bool CLAHE::ProcessImpl()
{
    if(!_clahe || clip_limit_param.modified || grid_size_param.modified)
    {
        _clahe = cv::cuda::createCLAHE(clip_limit, cv::Size(grid_size, grid_size));
        clip_limit_param.modified = false;
        grid_size_param.modified = false;
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