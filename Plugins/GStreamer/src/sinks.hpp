#pragma once

#include "gstreamer.hpp"
#include "EagleLib/Nodes/IFrameGrabber.hpp"

namespace EagleLib
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS tcpserver: public gstreamer_sink_base
        {
            bool _initialized;
        public:
            tcpserver();
            ~tcpserver();
            virtual void NodeInit(bool firstInit);
            virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream &stream);
        };
    }

    class PLUGIN_EXPORTS BufferedHeartbeatRtsp: public Nodes::FrameGrabberBuffered, public gstreamer_src_base
    {
    public:
        virtual void NodeInit(bool firstInit);

    protected:
        
    };


}