#include "Blur.hpp"
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include "Aquila/Nodes/NodeInfo.hpp"

using namespace aq;
using namespace aq::Nodes;

bool MedianBlur::ProcessImpl()
{
    if(!_median_filter || window_size_param._modified || partition_param._modified)
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
