#include "Blur.hpp"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/core/CvContext.hpp>
#include <opencv2/imgproc.hpp>
namespace aq
{
namespace nodes
{
template <>
bool GaussianBlur::processImpl(mo::Context* ctx)
{
    cv::Mat blurred;
    cv::Mat in = input->getMat(ctx);
    cv::GaussianBlur(in, blurred, {kerenl_size, kerenl_size}, sigma, sigma);
    output_param.updateData(blurred, mo::tag::_param = input_param);
    return true;
}

bool GaussianBlur::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}
}
}
using namespace aq::nodes;
MO_REGISTER_CLASS(GaussianBlur)
