#pragma once

#include "Aquila/Nodes/IFrameGrabber.hpp"
#include "gstreamer.hpp"

namespace aq
{
    class PLUGIN_EXPORTS chunked_file_sink: virtual public gstreamer_src_base, virtual public Nodes::FrameGrabberBuffered
    {
    protected:
        GstElement* _filesink;
    public:
        MO_DERIVE(chunked_file_sink, Nodes::FrameGrabberBuffered)
            PARAM(size_t, chunk_size, 10 * 1024 * 1024);
        MO_END;

        static int CanLoadDocument(const std::string& doc);
        static int LoadTimeout();

        virtual bool LoadFile(const std::string& file_path);
        virtual long long GetNumFrames();
        virtual rcc::shared_ptr<aq::ICoordinateManager> GetCoordinateManager();
        //virtual void Init(bool firstInit);
        virtual GstFlowReturn on_pull();
        bool ProcessImpl();
        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
    };

    class PLUGIN_EXPORTS JpegKeyframer: virtual public gstreamer_src_base, virtual public Nodes::FrameGrabberBuffered
    {
    public:
        MO_DERIVE(JpegKeyframer, Nodes::FrameGrabberBuffered);
            PROPERTY(long long, keyframe_count, 0);
        MO_END;
        static int CanLoadDocument(const std::string& doc);
        static int LoadTimeout();
        TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        long long GetNumFrames();
        long long GetFrameNum();
        bool LoadFile(const std::string& file_path);
        GstFlowReturn on_pull();
        rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
    protected:
        bool ProcessImpl();
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
