#include "IRosMessageReader.hpp"
#include "sensor_msgs/Image.h"
#include "ros/ros.h"
#include "ros/topic.h"
#include "MessageReaderInfo.hpp"
#include "RosInterface.hpp"
#include <cv_bridge/cv_bridge.h>
#include <Aquila/SyncedMemory.h>

class MessageReaderImage: public ros::IMessageReader
{
public:
    MO_DERIVE(MessageReaderImage,ros::IMessageReader)
        OUTPUT(aq::SyncedMemory, image, {})
    MO_END;

    static int CanHandleTopic(const std::string& topic)
    {
        ros::master::V_TopicInfo ti;
        if(!ros::master::getTopics(ti))
            return 0;
        for(ros::master::V_TopicInfo::iterator it = ti.begin(); it != ti.end(); ++it)
        {
            if(it->name == topic)
            {
                if(it->datatype == "sensor_msgs/Image")
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
        if(!ros::master::getTopics(ti))
            return;
        for(ros::master::V_TopicInfo::iterator it = ti.begin(); it != ti.end(); ++it)
        {
            if(it->datatype == "sensor_msgs/Image")
            {
                topics.push_back(it->name);
            }
        }
    }

    bool Subscribe(const std::string& topic)
    {
        _sub = aq::RosInterface::Instance()->nh()->subscribe<sensor_msgs::Image>(topic, 5,
                boost::bind(&MessageReaderImage::imageCb, this, _1));
        subscribed_topic = topic;
        return true;
    }

    bool ProcessImpl()
    {
        return true;
    }

    void imageCb(sensor_msgs::Image::ConstPtr msg)
    {
        cv_bridge::CvImageConstPtr img = cv_bridge::toCvCopy(msg, msg->encoding);
        cv::Mat i = img->image;
        image_param.UpdateData(i, mo::tag::_timestamp = mo::second * msg->header.stamp.toSec(),
                               mo::tag::_frame_number = msg->header.seq, _ctx);
    }

    ros::Subscriber _sub;
};

MO_REGISTER_CLASS(MessageReaderImage)
