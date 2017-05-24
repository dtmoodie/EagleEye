#pragma once
#include "ROSExport.hpp"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include "IRosMessageReader.hpp"

namespace aq
{
    namespace Nodes
    {
        class ROS_EXPORT RosSubscriber : public IFrameGrabber
        {
        public:
            static std::vector<std::string> ListLoadablePaths();
            static int CanLoadDocument(const std::string& topic);
            static int LoadTimeout(){return 10000;}
            MO_DERIVE(RosSubscriber, IFrameGrabber)
            MO_END
            bool Load(std::string file_path);

            void AddComponent(rcc::weak_ptr<Algorithm> component);
            void NodeInit(bool firstInit);
        protected:
            std::vector<rcc::shared_ptr<ros::IMessageReader>> _readers;
            bool ProcessImpl();
        };
    }
}
