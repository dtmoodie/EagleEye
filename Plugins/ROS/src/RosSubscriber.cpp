#include "RosSubscriber.hpp"
#include <Aquila/Nodes/FrameGrabberInfo.hpp>
#include "RosInterface.hpp"
#include <ros/ros.h>

using namespace aq;
using namespace aq::Nodes;

std::vector<std::string> RosSubscriber::ListLoadablePaths()
{
    return ros::IMessageReader::ListSubscribableTopics();
}
int RosSubscriber::CanLoadDocument(const std::string& topic)
{
    return ros::IMessageReader::CanLoadTopic(topic);
}

bool RosSubscriber::Load(std::string file_path)
{
    for(auto reader : _readers)
    {
        if(reader->subscribed_topic == file_path)
        {
            return true;
        }
    }
    auto reader = ros::IMessageReader::create(file_path);
    if(reader)
    {
        _readers.push_back(reader);
        this->AddComponent(reader);
        return true;
    }
    return false;
}


void RosSubscriber::AddComponent(rcc::weak_ptr<Algorithm> component)
{
    auto typed = component.DynamicCast<ros::IMessageReader>();
    if(typed)
    {
        _readers.push_back(typed);
        Algorithm::AddComponent(component);
    }
}
void RosSubscriber::NodeInit(bool firstInit)
{
    aq::RosInterface::Instance();
}

bool RosSubscriber::ProcessImpl()
{
    Node* This = this;
    sig_node_updated(This);
    sig_update();
    this->_modified = true;
    ros::spinOnce();
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    return true;
}

MO_REGISTER_CLASS(RosSubscriber)
