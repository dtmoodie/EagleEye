#include <Aquila/types/SyncedImage.hpp>

#include "RosPublisher.hpp"
#include "RosInterface.hpp"

#include "Aquila/nodes/NodeInfo.hpp"

#include <MetaObject/params/TMultiSubscriber.hpp>
#include <MetaObject/params/TypeSelector.hpp>

#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/Image.h"

#include <ros/node_handle.h>

using namespace aq;
using namespace aq::nodes;
namespace aqros
{

void ImagePublisher::nodeInit(bool firstInit)
{
}

sensor_msgs::Image::Ptr copyToMsg(cv::Mat mat)
{
    sensor_msgs::Image::Ptr msg = boost::make_shared<sensor_msgs::Image>();
    msg->encoding = "bgr8"; // mer make function to convert cv type
    msg->header = std_msgs::Header();
    msg->data.resize(mat.rows * mat.cols * mat.channels());
    cv::Mat wrapped(mat.rows, mat.cols, mat.type(), msg->data.data());
    mat.copyTo(wrapped);
    msg->width = mat.cols;
    msg->height = mat.rows;
    msg->step = mat.cols * 3;
    return msg;
}

bool ImagePublisher::processImpl()
{
    auto ts = input_param.getNewestTimestamp();
    // remove once TMultiInput time syncing is fixed
    if (m_prev_time && ts)
    {
        if (*m_prev_time == *ts)
        {
            return true;
        }
    }
    m_prev_time = ts;
    std_msgs::Header header;
    if (ts)
    {
        header.stamp.fromNSec(std::chrono::duration_cast<std::chrono::nanoseconds>(ts->time_since_epoch()).count());
    }
    std::shared_ptr<mo::IAsyncStream> stream = this->getStream();
    if (auto in = mo::get<const SyncedImage*>(input))
    {
        if (!_image_publisher)
        {
            _image_publisher = RosInterface::Instance()->nh()->advertise<sensor_msgs::Image>(topic_name, 1);
        }

        // Check where the data resides
        SyncedMemory::SyncState state = in->state();
        // Download to the CPU
        // TODO custom serializer
        cv::Mat h_input = in->getMat(stream.get());
        // If data is already on the cpu, we don't need to wait for the async download to finish
        if (state < SyncedMemory::SyncState::DEVICE_UPDATED)
        {
            // data already updated on cpu
            auto msg = copyToMsg(h_input);
            msg->header = header;
            _image_publisher.publish(msg);
            ros::spinOnce();
        }
        else
        {
            // data on gpu, do async publish
            auto tmp = *in;
            stream->pushWork(
                [header,tmp, h_input, this](mo::IAsyncStream*) -> void {
                    // This code is executed on cpu thead id after the download is complete
                    auto msg = copyToMsg(h_input);
                    msg->header = header;
                    _image_publisher.publish(msg);
                });
        }
    }
    else
    {
        if (const aq::CompressedImage* in = mo::get<const aq::CompressedImage*>(input))
        {
            if (!_image_publisher)
            {
                _image_publisher = RosInterface::Instance()->nh()->advertise<sensor_msgs::CompressedImage>(
                    topic_name + "/compressed", 1);
            }
            sensor_msgs::CompressedImage::Ptr msg = boost::make_shared<sensor_msgs::CompressedImage>();
            const auto data = in->getData();
            msg->data.resize(data->size());
            msg->format = "jpeg";
            msg->header = header;
            auto host_view = data->host(stream.get());
            // TODO custom serializer
            memcpy(msg->data.data(), host_view.data(), host_view.size());
            _image_publisher.publish(msg);
        }
    }
    return false;
}
}
using namespace aqros;
MO_REGISTER_CLASS(ImagePublisher)
