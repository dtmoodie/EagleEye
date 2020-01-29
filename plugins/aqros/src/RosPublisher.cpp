#include "RosPublisher.hpp"
#include "RosInterface.hpp"

#include "Aquila/nodes/NodeInfo.hpp"
#include "Aquila/utilities/cuda/CudaCallbacks.hpp"
#include <MetaObject/params/TMultiInput-inl.hpp>
#include <MetaObject/params/TypeSelector.hpp>

#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/Image.h"

#include <ros/node_handle.h>

using namespace aq;
using namespace aq::nodes;

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
    auto ts = input_param.getTimestamp();
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
    if (auto in = mo::get<const SyncedMemory*>(input))
    {
        if (!_image_publisher)
        {
            _image_publisher = RosInterface::Instance()->nh()->advertise<sensor_msgs::Image>(topic_name, 1);
        }

        // Check where the data resides
        SyncedMemory::SYNC_STATE state = in->getSyncState();
        // Download to the CPU
        cv::Mat h_input = in->getMat(stream());
        // If data is already on the cpu, we don't need to wait for the async download to finish
        if (state < SyncedMemory::DEVICE_UPDATED)
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
            size_t id = mo::getThisThread();
            aq::cuda::enqueue_callback_async(
                [header,tmp, h_input, this]() -> void {
                    // This code is executed on cpu thead id after the download is complete
                    auto msg = copyToMsg(h_input);
                    msg->header = header;
                    _image_publisher.publish(msg);
                },
                id,
                stream());
        }
    }
    else
    {
        if (const types::CompressedImage* in = mo::get<const types::CompressedImage*>(input))
        {
            if (!_image_publisher)
            {
                _image_publisher = RosInterface::Instance()->nh()->advertise<sensor_msgs::CompressedImage>(
                    topic_name + "/compressed", 1);
            }
            sensor_msgs::CompressedImage::Ptr msg = boost::make_shared<sensor_msgs::CompressedImage>();
            msg->data.resize(in->data.cols);
            msg->format = "jpeg";
            msg->header = header;

            memcpy(msg->data.data(), in->data.data, in->data.cols);
            _image_publisher.publish(msg);
        }
    }
    return false;
}

MO_REGISTER_CLASS(ImagePublisher)
