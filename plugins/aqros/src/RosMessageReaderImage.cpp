#include <Aquila/types/SyncedImage.hpp>

#include "IRosMessageReader.hpp"
#include "MessageReaderInfo.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include "RosInterface.hpp"

#include "ros/ros.h"
#include "ros/topic.h"

#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"

#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>

#include <opencv2/cudaimgproc.hpp>

#include <cuda_runtime_api.h>

template <class T>
struct StlAllocator
{
    StlAllocator()
    {
        mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::current();
        m_allocator = stream->hostAllocator();
        MO_ASSERT(m_allocator != nullptr);
    }
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    template <class U>
    struct rebind
    {
        typedef std::allocator<U> other;
    };
    pointer allocate(size_type n, std::allocator<void>::const_pointer /*hint*/ = 0)
    {
        pointer out;
        out = ct::ptrCast<T>(m_allocator->allocate(n * sizeof(T), sizeof(T)));
        return out;
    }

    void deallocate(T* p, std::size_t n) { m_allocator->deallocate(p, n * sizeof(T)); }

    mo::Allocator::Ptr_t m_allocator;
};

typedef sensor_msgs::Image_<StlAllocator<void>> PinnedImage;

class MessageReaderImage : public ros::IMessageReader
{
  public:
    MO_DERIVE(MessageReaderImage, ros::IMessageReader)
        OUTPUT(aq::SyncedImage, image, {})
    MO_END;

    static int CanHandleTopic(const std::string& topic)
    {
        aq::RosInterface::Instance();
        ros::master::V_TopicInfo ti;
        if (!ros::master::getTopics(ti))
            return 0;
        for (ros::master::V_TopicInfo::iterator it = ti.begin(); it != ti.end(); ++it)
        {
            if (it->name == topic)
            {
                if (it->datatype == "sensor_msgs/Image")
                {
                    return 10;
                }
            }
        }
        return 0;
    }

    static void ListTopics(std::vector<std::string>& topics)
    {
        aq::RosInterface::Instance();
        ros::master::V_TopicInfo ti;
        if (!ros::master::getTopics(ti))
            return;
        for (ros::master::V_TopicInfo::iterator it = ti.begin(); it != ti.end(); ++it)
        {
            if (it->datatype == "sensor_msgs/Image")
            {
                topics.push_back(it->name);
            }
        }
    }

    bool subscribe(const std::string& topic) override
    {
        _sub = aq::RosInterface::Instance()->nh()->subscribe<PinnedImage>(
            topic, 5, boost::bind(&MessageReaderImage::imageCb, this, _1));
        subscribed_topic = topic;
        return true;
    }

    bool processImpl() override { return true; }
    int cvBayerCode(const std::string& enc)
    {
        if (enc == sensor_msgs::image_encodings::BAYER_BGGR8 || enc == sensor_msgs::image_encodings::BAYER_BGGR16)
        {
            return cv::COLOR_BayerBG2BGR;
        }
        else if (enc == sensor_msgs::image_encodings::BAYER_RGGB8 || enc == sensor_msgs::image_encodings::BAYER_RGGB16)
        {
            return cv::COLOR_BayerRG2BGR;
        }
        else if (enc == sensor_msgs::image_encodings::BAYER_GBRG8 || enc == sensor_msgs::image_encodings::BAYER_GBRG16)
        {
            return cv::COLOR_BayerGB2BGR;
        }
        else if (enc == sensor_msgs::image_encodings::BAYER_GRBG8 || enc == sensor_msgs::image_encodings::BAYER_GRBG16)
        {
            return cv::COLOR_BayerGR2BGR;
        }
        else
        {
            throw std::runtime_error("Invalid bayer pattern " + enc);
        }
        return 0;
    }

    void imageCb(PinnedImage::ConstPtr msg)
    {
        std::string enc = msg->encoding.c_str();
        const int num_channels = sensor_msgs::image_encodings::numChannels(enc);

        const aq::DataFlag depth =
            sensor_msgs::image_encodings::bitDepth(enc) == 8 ? ct::value(aq::DataFlag::kUINT8) : ct::value(aq::DataFlag::kUINT16);

        aq::PixelType pixel_type;
        pixel_type.data_type = depth;
        if (enc.find("bgr") == 0)
        {
            pixel_type.pixel_format = aq::PixelFormat::kBGR;
            MO_ASSERT_EQ(num_channels, 3);
        }
        else if (enc.find("rgb") == 0)
        {
            pixel_type.pixel_format = aq::PixelFormat::kRGB;
            MO_ASSERT_EQ(num_channels, 3);
        }
        else if (enc.find("mono") == 0)
        {
            pixel_type.pixel_format = aq::PixelFormat::kGRAY;
        }
        else if (enc.find("bayer") == 0)
        {
            // TODO
            THROW(debug, "Unsupported image format");
        }

        auto stream = this->getStream();
        aq::Shape<2> shape(msg->height, msg->width);

        aq::SyncedImage image(shape,
                              pixel_type,
                              msg->data.data(),
                              std::shared_ptr<const void>(msg->data.data(), [msg](const void*) {}),
                              stream);

        this->image.publish(std::move(image),
                            mo::tags::timestamp = mo::second * msg->header.stamp.toSec(),
                            mo::tags::fn = msg->header.seq);
    }

    ros::Subscriber _sub;
};

MO_REGISTER_CLASS(MessageReaderImage)
