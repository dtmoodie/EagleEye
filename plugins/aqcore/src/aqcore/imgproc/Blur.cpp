#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "Blur.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>

using namespace aq;
using namespace aq::nodes;

bool MedianBlur::processImpl()
{
    if (!_median_filter || window_size_param.modified() || partition_param.modified())
    {
        _median_filter = cv::cuda::createMedianFilter(input->getDepth(), window_size, partition);
    }
    cv::cuda::GpuMat output;
    if (input->getChannels() != 1 && false)
    {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(input->getGpuMat(stream()), channels, stream());
        std::vector<cv::cuda::GpuMat> blurred(channels.size());
        for (int i = 0; i < channels.size(); ++i)
        {
            _median_filter->apply(channels[i], blurred[i], stream());
        }
        cv::cuda::merge(blurred, output, stream());
    }
    else
    {
        _median_filter->apply(input->getGpuMat(stream()), output, stream());
    }
    output_param.updateData(output, input_param.getTimestamp(), _ctx.get());

    return true;
}

MO_REGISTER_CLASS(MedianBlur)

#endif