#pragma once
#include "Aquila/nodes/Node.hpp"
#include "IRosMessageWriter.hpp"
#include "ROSExport.hpp"
#include "ros/publisher.h"
#include <Aquila/types/SyncedMemory.hpp>
namespace aq
{
    namespace nodes
    {
        class ROS_EXPORT RosPublisher : public Node
        {
          public:
            MO_DERIVE(RosPublisher, Node)

            MO_END
          protected:
        };
        class ROS_EXPORT ImagePublisher : public Node
        {
          public:
            MO_DERIVE(ImagePublisher, Node)
            INPUT(SyncedMemory, input, nullptr)
            PARAM(std::string, topic_name, "image")
            MO_END
            void nodeInit(bool firstInit);

          protected:
            bool processImpl();
            ros::Publisher _image_publisher;
        };
    }
}
