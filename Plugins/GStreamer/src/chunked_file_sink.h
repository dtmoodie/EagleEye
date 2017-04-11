#pragma once

#include "Aquila/Nodes/IFrameGrabber.hpp"
#include "gstreamer.hpp"

namespace aq
{
    class PLUGIN_EXPORTS chunked_file_sink: 
        virtual public gstreamer_src_base, 
        virtual public Nodes::IGrabber
    {
    public:
        static int CanLoad(const std::string& doc);
        static int Timeout();
        MO_DERIVE(chunked_file_sink, Nodes::IGrabber)
            PARAM(size_t, chunk_size, 10 * 1024 * 1024)
        MO_END;
        virtual bool Load(const std::string& file_path);
        virtual GstFlowReturn on_pull();
    protected:
        bool Grab(){return true;}
        GstElement* _filesink;
    };

    class PLUGIN_EXPORTS JpegKeyframer:
        virtual public gstreamer_src_base,
        virtual public Nodes::IGrabber
    {
    public:
        MO_DERIVE(JpegKeyframer, Nodes::IGrabber)
            PROPERTY(long long, keyframe_count, 0)
        MO_END;
        static int CanLoad(const std::string& doc);
        static int Timeout();
        bool Load(const std::string& file_path);
        GstFlowReturn on_pull();
    protected:
        bool Grab(){return true;}
    };
    namespace Nodes
    {
    class GstreamerSink: virtual public gstreamer_sink_base
    {
    public:
        MO_DERIVE(GstreamerSink, gstreamer_sink_base)
            INPUT(SyncedMemory, image, nullptr);
            PARAM(std::string, pipeline, "");
        MO_END;
    protected:
        bool ProcessImpl();
    };
    }
}
