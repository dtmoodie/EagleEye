#pragma once
#include "IRosMessageReader.hpp"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

namespace aq
{
    namespace nodes
    {
        class RosSubscriber : public IFrameGrabber
        {
          public:
            static std::vector<std::string> listLoadablePaths();
            static int canLoadPath(const std::string& topic);
            static int loadTimeout() { return 10000; }

            MO_DERIVE(RosSubscriber, IFrameGrabber)
            MO_END;

            bool loadData(std::string file_path) override;

            void addComponent(const rcc::weak_ptr<IAlgorithm>& component) override;
            void nodeInit(bool firstInit) override;

          protected:
            bool processImpl() override;

            std::vector<rcc::shared_ptr<ros::IMessageReader>> _readers;
        };
    }
}
