#include "chunked_file_sink.h"
#include <EagleLib/Nodes/FrameGrabberInfo.hpp>
#include <EagleLib/ICoordinateManager.h>
#include <gst/base/gstbasesink.h>
using namespace EagleLib;


int chunked_file_sink::CanLoadDocument(const std::string& document)
{
    return 0; // Currently needs to be manually specified
}
int chunked_file_sink::LoadTimeout()
{
    return 3000;
}



GstFlowReturn chunked_file_sink::on_pull()
{
    GstSample *sample = gst_base_sink_get_last_sample(GST_BASE_SINK(_appsink));
    //g_signal_emit_by_name(_appsink, "pull-sample", &sample, NULL);
    if(sample)
    {
        GstBuffer *buffer;
        GstCaps *caps;
        GstStructure *s;
        GstMapInfo map;
        caps = gst_sample_get_caps (sample);
        if (!caps) 
        {
            LOG(debug) << "could not get sample caps";
            return GST_FLOW_OK;
        }
        s = gst_caps_get_structure (caps, 0);
        gint width, height;
        gboolean res; 
        res = gst_structure_get_int (s, "width", &width);
        res |= gst_structure_get_int (s, "height", &height);
        const gchar* format = gst_structure_get_string(s, "format");
        if (!res) 
        {
            LOG(debug) << "could not get snapshot dimension\n";
            return GST_FLOW_OK;
        }
        buffer = gst_sample_get_buffer (sample);
        if (gst_buffer_map (buffer, &map, GST_MAP_READ))
        {
            cv::Mat mapped(height, width, CV_8UC3);
            memcpy(mapped.data, map.data, map.size);

        }
        gst_sample_unref (sample);
    }
    return GST_FLOW_OK;
}
bool chunked_file_sink::LoadFile(const std::string& file_path)
{
    if(gstreamer_src_base::create_pipeline(file_path))
    {
        _filesink = gst_bin_get_by_name(GST_BIN(_pipeline), "filesink0");
        if(_filesink)
        {
            this->start_pipeline();
            return true;
        }
    }
    return false;
}
TS<SyncedMemory> chunked_file_sink::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return TS<SyncedMemory>();
}
long long chunked_file_sink::GetNumFrames()
{
    return -1;
}

rcc::shared_ptr<EagleLib::ICoordinateManager> chunked_file_sink::GetCoordinateManager()
{
    return rcc::shared_ptr<EagleLib::ICoordinateManager>();
}

bool chunked_file_sink::ProcessImpl()
{
    return false;
}

MO_REGISTER_CLASS(chunked_file_sink);

int JpegKeyframer::CanLoadDocument(const std::string& doc)
{
    if(doc.find("http") != std::string::npos && doc.find("mjpg") != std::string::npos)
    {
        return 10;
    }
    return 0;
}

int JpegKeyframer::LoadTimeout()
{
    return 10000;
}

bool JpegKeyframer::LoadFile(const std::string& file_path)
{
    std::stringstream pipeline;
    pipeline << "souphttpsrc location=" << file_path;
    pipeline << " ! multipartdemux ! appsink name=mysink";
    
    if(this->create_pipeline(pipeline.str()))
    {
        if(this->set_caps("image/jpeg"))
        {
            this->start_pipeline();
            return true;
        }
    }
    return false;
}

GstFlowReturn JpegKeyframer::on_pull()
{
    GstSample *sample = gst_base_sink_get_last_sample(GST_BASE_SINK(_appsink));
    if (sample)
    {
        GstBuffer *buffer;
        GstCaps *caps;
        GstStructure *s;
        GstMapInfo map;
        caps = gst_sample_get_caps(sample);
        if (!caps)
        {
            LOG(debug) << "could not get sample caps";
            return GST_FLOW_OK;
        }
        ++keyframe_count;
    }
    return GST_FLOW_OK;
}
    
bool JpegKeyframer::ProcessImpl()
{
    return true;
}
TS<SyncedMemory> JpegKeyframer::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return TS<SyncedMemory>();
}
long long JpegKeyframer::GetNumFrames()
{
    return -1;
}
long long JpegKeyframer::GetFrameNum()
{
    return keyframe_count;
}
rcc::shared_ptr<ICoordinateManager> JpegKeyframer::GetCoordinateManager()
{
    return rcc::shared_ptr<ICoordinateManager>();
}
MO_REGISTER_CLASS(JpegKeyframer);

using namespace EagleLib::Nodes;
bool GstreamerSink::ProcessImpl()
{
    if(pipeline_param.modified && !image->empty())
    {
        this->cleanup();
        if(!this->create_pipeline(pipeline))
        {
            LOG(warning) << "Unable to create pipeline " << pipeline;
            return false;
        }
        if(!this->set_caps(image->GetSize(), image->GetChannels(), image->GetDepth()))
        {
            LOG(warning) << "Unable to set caps on pipeline";
            return false;
        }
        if(!this->start_pipeline())
        {
            LOG(warning) << "Unable to start pipeline " << pipeline;
            return false;
        }
        pipeline_param.modified = false;
    }
    if(_source && _feed_enabled)
    {
        PushImage(*image, Stream());
        return true;
    }
    return false;
}
MO_REGISTER_CLASS(GstreamerSink)
