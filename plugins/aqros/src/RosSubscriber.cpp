#include "RosSubscriber.hpp"
#include "RosInterface.hpp"
#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <ros/ros.h>

using namespace aq;
using namespace aq::nodes;

std::vector<std::string> RosSubscriber::listLoadablePaths()
{
    return ros::IMessageReader::ListSubscribableTopics();
}
int RosSubscriber::canLoadPath(const std::string& topic)
{
    return ros::IMessageReader::CanLoadTopic(topic);
}

bool RosSubscriber::loadData(std::string file_path)
{
    for (auto reader : _readers)
    {
        if (reader->subscribed_topic == file_path)
        {
            return true;
        }
    }
    auto reader = ros::IMessageReader::create(file_path);
    if (reader)
    {
        _readers.push_back(reader);
        this->addComponent(reader);
        return true;
    }
    return false;
}

void RosSubscriber::addComponent(const rcc::weak_ptr<IAlgorithm>& component)
{
    auto typed = component.DynamicCast<ros::IMessageReader>();
    if (typed)
    {
        _readers.push_back(typed);
        Algorithm::addComponent(component);
    }
}
void RosSubscriber::nodeInit(bool /*firstInit*/)
{
    aq::RosInterface::Instance();
}

bool RosSubscriber::processImpl()
{
    Node* This = this;
    sig_node_updated(This);
    sig_update();
    this->setModified();
    ros::spinOnce();
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    return true;
}

MO_REGISTER_CLASS(RosSubscriber)
