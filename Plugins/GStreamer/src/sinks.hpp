#pragma once
#include "MetaObject/MetaObject.hpp"
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
            enum {
                None = -1
            };
            MO_DERIVE(tcpserver, gstreamer_sink_base)
                ENUM_PARAM(encoders, None);
                ENUM_PARAM(interfaces, None);
                INPUT(SyncedMemory, image, nullptr);
            MO_END;
            tcpserver();
            ~tcpserver();
            virtual void NodeInit(bool firstInit);
            bool ProcessImpl();
        };

        class PLUGIN_EXPORTS BufferedHeartbeatRtsp : public FrameGrabberBuffered, public gstreamer_src_base
        {
        public:
            virtual void NodeInit(bool firstInit);
        protected:
        };

        class PLUGIN_EXPORTS JPEGSink: public Node, public gstreamer_src_base
        {
        public:
            JPEGSink();
            MO_DERIVE(JPEGSink, Node)
                PARAM(std::string, gstreamer_pipeline, "");
                OUTPUT(cv::Mat, jpeg_buffer, cv::Mat());
                OUTPUT(cv::Mat, decoded, cv::Mat());
            MO_END;
        protected:
            bool ProcessImpl();
            virtual GstFlowReturn on_pull();
            cv::Mat decode_buffer;
            mo::Context gstreamer_context;
        };
    }
}