#include "Blur.hpp"
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include "EagleLib/Nodes/NodeInfo.hpp"

using namespace EagleLib;
using namespace EagleLib::Nodes;

bool MedianBlur::ProcessImpl()
{
    if(!_median_filter || window_size_param.modified || partition_param.modified)
    {
        _median_filter = cv::cuda::createMedianFilter(input->GetDepth(), window_size, partition);
    }
    cv::cuda::GpuMat output;
    if(input->GetChannels() != 1 && false)
    {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(input->GetGpuMat(Stream()), channels, Stream());
        std::vector<cv::cuda::GpuMat> blurred(channels.size());
        for(int i = 0; i < channels.size(); ++i)
        {
            _median_filter->apply(channels[i],blurred[i], Stream());
        }
        cv::cuda::merge(blurred, output, Stream());
    }else
    {
        _median_filter->apply(input->GetGpuMat(Stream()), output, Stream());
    }
    output_param.UpdateData(output, input_param.GetTimestamp(), _ctx);

    return true;
}

MO_REGISTER_CLASS(MedianBlur)
