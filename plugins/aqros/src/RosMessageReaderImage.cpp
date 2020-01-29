#include "IRosMessageReader.hpp"
#include "MessageReaderInfo.hpp"
#include "MetaObject/core/detail/AllocatorImpl.hpp"
#include "MetaObject/core/detail/MemoryStack.hpp"
#include "RosInterface.hpp"
#include "ros/ros.h"
#include "ros/topic.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <opencv2/cudaimgproc.hpp>

template <class T>
struct StlAllocator
{
    typedef T value_type;
    typedef T* pointer;
    typedef std::size_t size_type;
    template <class U>
    struct rebind
    {
        typedef std::allocator<U> other;
    };
    pointer allocate(size_type n, std::allocator<void>::const_pointer /*hint*/ = 0)
    {
        pointer out;
        mo::CpuMemoryStack::globalInstance()->allocate(&out, n * sizeof(T), sizeof(T));
        return out;
    }

    void deallocate(T* p, std::size_t n) { mo::CpuMemoryStack::globalInstance()->deallocate(p, n); }
};

typedef sensor_msgs::Image_<StlAllocator<void>> PinnedImage;

class MessageReaderImage : public ros::IMessageReader
{
  public:
    MO_DERIVE(MessageReaderImage, ros::IMessageReader)
        OUTPUT(aq::SyncedMemory, image, {})
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
        cv::cuda::GpuMat gpu_buf;
        std::string enc = msg->encoding.c_str();
        int c = sensor_msgs::image_encodings::numChannels(enc);
        int depth = sensor_msgs::image_encodings::bitDepth(enc) == 8 ? CV_8U : CV_16U;
        gpu_buf.create(msg->height, msg->width, CV_MAKE_TYPE(depth, c));
        size_t src_stride = (depth == CV_8U ? 1 : 2) * msg->width * c;
        cudaMemcpy2DAsync(gpu_buf.datastart,
                          gpu_buf.step,
                          (const void*)msg->data.data(),
                          src_stride,
                          src_stride,
                          msg->height,
                          cudaMemcpyHostToDevice,
                          _ctx->getCudaStream());
        if (c == 1 && sensor_msgs::image_encodings::isBayer(enc))
        {
            cv::cuda::GpuMat color;
            cv::cuda::demosaicing(gpu_buf, color, cvBayerCode(enc), -1, _ctx->getStream());
            gpu_buf = color;
        }
        image_param.updateData(gpu_buf,
                               mo::tag::_timestamp = mo::second * msg->header.stamp.toSec(),
                               mo::tag::_frame_number = msg->header.seq,
                               _ctx);
    }

    ros::Subscriber _sub;
};

MO_REGISTER_CLASS(MessageReaderImage)
