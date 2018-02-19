#pragma once

#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "gstreamer.hpp"

namespace aq
{
    class aqgstreamer_EXPORT chunked_file_sink:
        virtual public gstreamer_src_base, 
        virtual public nodes::IGrabber
    {
    public:
        static int canLoad(const std::string& doc);
        static int loadTimeout();
        MO_DERIVE(chunked_file_sink, nodes::IGrabber)
            PARAM(size_t, chunk_size, 10 * 1024 * 1024)
        MO_END;
        virtual bool loadData(const std::string& file_path);
        virtual GstFlowReturn on_pull();
    protected:
        bool grab(){return true;}
        GstElement* _filesink;
    };

    class aqgstreamer_EXPORT JpegKeyframer:
        virtual public gstreamer_src_base,
        virtual public nodes::IGrabber
    {
    public:
        MO_DERIVE(JpegKeyframer, nodes::IGrabber)
            PROPERTY(long long, keyframe_count, 0)
        MO_END;
        static int canLoad(const std::string& doc);
        static int loadTimeout();
        bool loadData(const std::string& file_path);
        GstFlowReturn on_pull();
    protected:
        bool grab(){return true;}
    };
    namespace nodes
    {
    class aqgstreamer_EXPORT GstreamerSink: virtual public gstreamer_sink_base
    {
    public:
        MO_DERIVE(GstreamerSink, gstreamer_sink_base)
            INPUT(SyncedMemory, image, nullptr);
            PARAM(std::string, pipeline, "");
        MO_END;
    protected:
        bool processImpl();
    };
    }
}
