#pragma once
#include "GStreamerExport.hpp"
#include "MetaObject/MetaObject.hpp"
#include "gstreamer.hpp"
#include "Aquila/Nodes/IFrameGrabber.hpp"

namespace aq
{
    namespace Nodes
    {
        class GStreamer_EXPORT tcpserver: public gstreamer_sink_base
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

        class GStreamer_EXPORT GStreamerSink: public gstreamer_sink_base
        {
        public:
            MO_DERIVE(GStreamerSink, gstreamer_sink_base)
                INPUT(SyncedMemory, image, nullptr)
                PARAM(std::string, gstreamer_pipeline, "")
            MO_END
        protected:
            bool ProcessImpl();
            bool _initialized = false;
        };

        class GStreamer_EXPORT BufferedHeartbeatRtsp : public FrameGrabberBuffered, public gstreamer_src_base
        {
        public:
            virtual void NodeInit(bool firstInit);
        protected:
        };

        class GStreamer_EXPORT JPEGSink: public Node, public gstreamer_src_base
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
