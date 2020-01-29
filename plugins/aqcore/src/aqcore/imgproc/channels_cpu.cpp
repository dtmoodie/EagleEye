#include "Channels.h"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>

namespace aq
{
namespace nodes
{
template <>
bool ConvertToGrey::processImpl(mo::Context* ctx)
{
    const auto& mat = input->getMat(ctx, 0);
    cv::Mat gray_mat;
    cv::cvtColor(mat, gray_mat, cv::COLOR_BGR2GRAY);
    grey_param.updateData(gray_mat, mo::tag::_param = input_param);
    return true;
}

bool ConvertToGrey::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

template <>
bool ConvertToHSV::processImpl(mo::Context* ctx)
{
    const cv::Mat& mat = input_image->getMat(ctx);
    cv::Mat hsv;
    cv::cvtColor(mat, hsv, cv::COLOR_BGR2HSV);
    hsv_image_param.updateData(hsv, input_image_param.getTimestamp(), this->_ctx.get());
    return true;
}

bool ConvertToHSV::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

template <>
bool SplitChannels::processImpl(mo::Context* ctx)
{
    std::vector<cv::Mat> _channels;
    const cv::Mat& mat = input->getMat(ctx);
    cv::split(mat, _channels);
    output_param.updateData(_channels, mo::tag::_param = input_param);
    return true;
}

bool SplitChannels::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

template <>
bool ConvertDataType::processImpl(mo::Context* ctx)
{
    const cv::Mat& mat = input->getMat(ctx);
    cv::Mat output;
    mat.convertTo(output, data_type.getValue());
    output_param.updateData(output, mo::tag::_param = input_param);
    return true;
}

bool ConvertDataType::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

template <>
bool Reshape::processImpl(mo::Context* ctx)
{
    const cv::Mat& mat = input_image->getMat(ctx);
    cv::Mat reshaped = mat.reshape(channels, rows);
    reshaped_image_param.updateData(reshaped, mo::tag::_param = input_image_param);
    return true;
}

bool Reshape::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

bool MergeChannels::processImpl()
{
    return false;
}

template <>
bool ConvertColorspace::processImpl(mo::Context* ctx)
{
    const cv::Mat& in = input_image->getMat(ctx);
    cv::Mat img;
    cv::cvtColor(in, img, conversion_code.getValue());
    output_image_param.updateData(img, mo::tag::_param = input_image_param);
    return true;
}

bool ConvertColorspace::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}

template <>
bool Magnitude::processImpl(mo::Context* ctx)
{
    // cv::Mat mag;
    // cv::magnitude(input_image->getMat(ctx), mag);
    // output_magnitude_param.updateData(mag, mo::tag::_param = input_image_param);
    return false;
}

bool Magnitude::processImpl()
{
    return nodeContextSwitch(this, _ctx.get());
}
}
}

using namespace aq::nodes;
MO_REGISTER_CLASS(ConvertToGrey)
MO_REGISTER_CLASS(ConvertToHSV)
MO_REGISTER_CLASS(ConvertDataType)
MO_REGISTER_CLASS(SplitChannels)
MO_REGISTER_CLASS(Reshape)
MO_REGISTER_CLASS(Magnitude)
