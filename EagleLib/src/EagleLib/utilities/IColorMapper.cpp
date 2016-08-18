#include "EagleLib/utilities/IColorMapper.hpp"
using namespace EagleLib;

IColorMapper::~IColorMapper()
{

}

cv::cuda::GpuMat IColorMapper::Apply(cv::cuda::GpuMat input, cv::InputArray mask, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat output;
    Apply(input, output, mask, stream);
    return output;
}

cv::Mat IColorMapper::Apply(cv::Mat input, cv::InputArray mask)
{
    cv::Mat output;
    Apply(input, output, mask);
    return output;
}