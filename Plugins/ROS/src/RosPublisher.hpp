#pragma once
#include "ROSExport.hpp"
#include "Aquila/nodes/Node.hpp"
#include "IRosMessageWriter.hpp"
#include "ros/publisher.h"
namespace aq
{
namespace Nodes
{
class ROS_EXPORT RosPublisher: public Node
{
public:
    MO_DERIVE(RosPublisher, Node)

    MO_END
protected:

};
class ROS_EXPORT ImagePublisher: public Node
{
public:
    MO_DERIVE(ImagePublisher, Node)
        INPUT(SyncedMemory, input, nullptr)
        PARAM(std::string, topic_name, "image")
    MO_END
    void NodeInit(bool firstInit);
protected:
    bool ProcessImpl();
    ros::Publisher _image_publisher;
};

}
}
