#ifndef AQGSTREAMER_SINKS_HPP
#define AQGSTREAMER_SINKS_HPP
#include "gstreamer.hpp"

#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "aqgstreamer/aqgstreamer_export.hpp"

namespace aqgstreamer
{

    class aqgstreamer_EXPORT TCPServer : public GstreamerSinkBase
    {
        bool _initialized = false;

      public:
        enum
        {
            None = -1
        };
        MO_DERIVE(TCPServer, GstreamerSinkBase)
            ENUM_PARAM(encoders, None);
            ENUM_PARAM(interfaces, None);
            INPUT(aq::SyncedImage, image);
        MO_END;

        void nodeInit(bool firstInit) override;
        bool processImpl() override;
    };

    class aqgstreamer_EXPORT JPEGSink : public aq::nodes::Node, public GstreamerSrcBase
    {
      public:
        JPEGSink();
        MO_DERIVE(JPEGSink, aq::nodes::Node)
            PARAM(std::string, pipeline, "");
            SOURCE(aq::CompressedImage, jpeg_buffer);
            SOURCE(aq::SyncedImage, decoded);
        MO_END;

      protected:
        bool processImpl() override;
        GstFlowReturn onPull(GstAppSink*) override;

        cv::Mat m_decode_buffer;
        std::shared_ptr<mo::IAsyncStream> m_gstreamer_stream;
    };
} // namespace aqgstreamer

#endif // AQGSTREAMER_SINKS_HPP
