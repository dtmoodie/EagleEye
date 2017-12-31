#pragma once
#include "IRosMessageReader.hpp"
#include "ROSExport.hpp"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

namespace aq
{
    namespace nodes
    {
        class ROS_EXPORT RosSubscriber : public IFrameGrabber
        {
          public:
            static std::vector<std::string> listLoadablePaths();
            static int canLoadPath(const std::string& topic);
            static int loadTimeout() { return 10000; }
            MO_DERIVE(RosSubscriber, IFrameGrabber)
            MO_END
            bool loadData(std::string file_path);

            void addComponent(rcc::weak_ptr<Algorithm> component);
            void nodeInit(bool firstInit);

          protected:
            std::vector<rcc::shared_ptr<ros::IMessageReader>> _readers;
            bool processImpl();
        };
    }
}
