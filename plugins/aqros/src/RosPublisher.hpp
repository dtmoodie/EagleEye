#pragma once
#include "Aquila/nodes/Node.hpp"
#include "IRosMessageWriter.hpp"
#include "ros/publisher.h"
#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <MetaObject/params/TMultiSubscriber.hpp>

namespace aqros
{

    class RosPublisher : public aq::nodes::Node
    {
      public:
        MO_DERIVE(RosPublisher, aq::nodes::Node)

        MO_END;

      protected:
    };

    class ImagePublisher : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ImagePublisher, aq::nodes::Node)
            MULTI_INPUT(input, aq::SyncedImage, aq::CompressedImage)
            PARAM(std::string, topic_name, "image")
        MO_END;
        void nodeInit(bool firstInit) override;

      protected:
        bool processImpl() override;
        ros::Publisher _image_publisher;
        // Hack for bug in MULTI_INPUT for now :/
        mo::OptionalTime m_prev_time;
    };

} // namespace aqros
