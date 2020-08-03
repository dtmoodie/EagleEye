#include "Channels.h"
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <Aquila/nodes/NodeInfo.hpp>

namespace aqcore
{
    template <>
    bool ConvertToGrey::processImpl(mo::IAsyncStream& stream)
    {
        const cv::Mat mat = input->getMat(&stream);
        cv::Mat gray_mat;
        cv::cvtColor(mat, gray_mat, cv::COLOR_BGR2GRAY);
        output.publish(gray_mat, mo::tags::param = &input_param, stream);
        return true;
    }

    bool ConvertToGrey::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }

    template <>
    bool ConvertToHSV::processImpl(mo::IAsyncStream& stream)
    {
        const cv::Mat mat = input->getMat(&stream);
        cv::Mat hsv;
        cv::cvtColor(mat, hsv, cv::COLOR_BGR2HSV);
        output.publish(hsv, mo::tags::param = &input_param, stream);
        return true;
    }

    bool ConvertToHSV::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }

    /*template <>
    bool SplitChannels::processImpl(mo::IAsyncStream& stream)
    {
        std::vector<cv::Mat> _channels;
        const cv::Mat mat = input->getMat(&stream);
        cv::split(mat, _channels);
        output.publish(_channels, mo::tags::param = &input_param);
        return true;
    }*/

    /*bool SplitChannels::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }*/

    template <>
    bool ConvertDataType::processImpl(mo::IAsyncStream& stream)
    {
        const cv::Mat mat = input->getMat(&stream);
        cv::Mat output;
        mat.convertTo(output, data_type.getValue());
        this->output.publish(output, mo::tags::param = &input_param);
        return true;
    }

    bool ConvertDataType::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }

    template <>
    bool Reshape::processImpl(mo::IAsyncStream& stream)
    {
        const cv::Mat mat = input->getMat(&stream);
        cv::Mat reshaped = mat.reshape(channels, rows);
        output.publish(reshaped, mo::tags::param = &input_param);
        return true;
    }

    bool Reshape::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }

    bool MergeChannels::processImpl() { return false; }

    template <>
    bool ConvertColorspace::processImpl(mo::IAsyncStream& stream)
    {
        const cv::Mat in = input->getMat(&stream);
        cv::Mat img;
        cv::cvtColor(in, img, conversion_code.getValue());
        output.publish(img, mo::tags::param = &input_param);
        return true;
    }

    bool ConvertColorspace::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }

    template <>
    bool Magnitude::processImpl(mo::IAsyncStream& stream)
    {
        // cv::Mat mag;
        // cv::magnitude(input_image->getMat(ctx), mag);
        // output_magnitude_param.updateData(mag, mo::tag::_param = input_image_param);
        return false;
    }

    bool Magnitude::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        return nodeStreamSwitch(this, *stream);
    }

} // namespace aqcore

using namespace aqcore;
MO_REGISTER_CLASS(ConvertToGrey)
MO_REGISTER_CLASS(ConvertToHSV)
MO_REGISTER_CLASS(ConvertDataType)
// MO_REGISTER_CLASS(SplitChannels)
MO_REGISTER_CLASS(Reshape)
MO_REGISTER_CLASS(Magnitude)
