#ifndef AQGSTREAMER_CHUNKED_FILES_SINK_HPP
#define AQGSTREAMER_CHUNKED_FILES_SINK_HPP

#include "gstreamer.hpp"

#include "Aquila/framegrabbers/IFrameGrabber.hpp"

namespace aqgstreamer
{
    class aqgstreamer_EXPORT ChunkedFileSink : public GstreamerSrcBase, virtual public aq::nodes::IGrabber
    {
      public:
        static int canLoad(const std::string& doc);
        static int loadTimeout();

        MO_DERIVE(ChunkedFileSink, aq::nodes::IGrabber)
            PARAM(size_t, chunk_size, 10 * 1024 * 1024)
        MO_END;

        bool loadData(const std::string& file_path) override;
        GstFlowReturn onPull(GstAppSink* appsink) override;

      protected:
        bool grab() override { return true; }
        GstElement* _filesink;
    };

    class aqgstreamer_EXPORT JpegKeyframer : public GstreamerSrcBase, public aq::nodes::IGrabber
    {
      public:
        MO_DERIVE(JpegKeyframer, aq::nodes::IGrabber)
            STATE(long long, keyframe_count, 0)
        MO_END;

        static int canLoad(const std::string& doc);
        static int loadTimeout();

        bool loadData(const std::string& file_path) override;
        GstFlowReturn onPull(GstAppSink* appsink) override;

      protected:
        bool grab() override { return true; }
    };

    class aqgstreamer_EXPORT GstreamerSink : virtual public GstreamerSinkBase
    {
      public:
        MO_DERIVE(GstreamerSink, GstreamerSinkBase)
            INPUT(aq::SyncedImage, image);
            PARAM(std::string, pipeline, "");
        MO_END;

      protected:
        bool processImpl() override;
    };

} // namespace aqgstreamer

#endif // AQGSTREAMER_CHUNKED_FILES_SINK_HPP
