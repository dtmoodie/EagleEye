#pragma once
#include "Aquila/nodes/Node.hpp"
#include "IRosMessageWriter.hpp"
#include "ros/publisher.h"
#include <Aquila/types/SyncedMemory.hpp>
namespace aq
{
    namespace nodes
    {
        class RosPublisher : public Node
        {
          public:
            MO_DERIVE(RosPublisher, Node)

            MO_END
          protected:
        };

        class ImagePublisher : public Node
        {
          public:
            MO_DERIVE(ImagePublisher, Node)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(std::string, topic_name, "image")
            MO_END
            virtual void nodeInit(bool firstInit) override;

          protected:
            virtual bool processImpl() override;
            ros::Publisher _image_publisher;
        };
    }
}
