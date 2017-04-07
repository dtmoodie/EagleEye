#pragma once
#include "ROSExport.hpp"
#include <Aquila/Nodes/IFrameGrabber.hpp>
#include "IRosMessageReader.hpp"

namespace aq
{
    namespace Nodes
    {
        class ROS_EXPORT RosSubscriber : public IFrameGrabber
        {
        public:
            static std::vector<std::string> ListLoadableDocuments();
            static int CanLoadDocument(const std::string& topic);
            MO_DERIVE(RosSubscriber, IFrameGrabber)
            MO_END
            bool LoadFile(const std::string& file_path);
            long long GetFrameNumber();
            long long GetNumFrames();
            TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
            TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
            TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);

            TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream);

            rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
            void AddComponent(rcc::weak_ptr<Algorithm> component);
            void NodeInit(bool firstInit);
        protected:
            std::vector<rcc::shared_ptr<ros::IMessageReader>> _readers;
            bool ProcessImpl();
        };
    }
}
