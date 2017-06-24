#include "sinks.hpp"
#include <gst/gst.h>
#include <Aquila/nodes/NodeInfo.hpp>
#include <gst/base/gstbasesink.h>
#include <opencv2/imgcodecs.hpp>
#include "glib_thread.h"
using namespace aq;
using namespace aq::nodes;

tcpserver::tcpserver():
    gstreamer_sink_base()
{
    _initialized = false;
    
}

tcpserver::~tcpserver()
{

}

void tcpserver::nodeInit(bool firstInit)
{
    if(firstInit)
    {
        encoders.addEnum(-1, "Select encoder");
    
        if(check_feature("matroskamux"))
        {
            if(check_feature("openh264enc"))
                encoders.addEnum(0, "openh264enc");
            if(check_feature("avenc_h264"))
                encoders.addEnum(1, "avenc_h264");
            if(check_feature("omxh264enc"))
                encoders.addEnum(2, "omxh264enc");
        }
        if(check_feature("webmmux"))
        {
            if(check_feature("omxvp8enc"))
                encoders.addEnum(3, "omxvp8enc");
            if(check_feature("vp8enc"))
                encoders.addEnum(4, "vp8enc");
        }
        encoders_param.emitUpdate();
        auto interfaces = get_interfaces();
        for(int i = 0; i < interfaces.size(); ++i)
        {
            this->interfaces.addEnum(i, interfaces[i]);
        }
        this->interfaces_param.emitUpdate();
    }
}

bool tcpserver::processImpl()
{
    if (!_initialized || encoders_param.modified() || interfaces_param.modified())
    {
        if (encoders.getValue() != -1)
        {
            std::string name = encoders.getEnum();
            std::stringstream ss;
            ss << "appsrc name=mysource ! videoconvert ! ";
            ss << name;
            if (name == "openh264enc" || name == "avenc_h264" || name == "omxh264enc")
            {
                ss << " ! matroskamux streamable=true ! tcpserversink host=";
            }
            else if (name == "omxvp8enc" || name == "vp8enc")
            {
                ss << " ! webmmux ! tcpserversink host=";
            }
            ss << interfaces.getEnum();
            ss << " port=8080";
            _initialized = create_pipeline(ss.str());
            if (_initialized)
            {
                encoders_param.modified(false);
                interfaces_param.modified(false);
            }
        }
    }
    if(_initialized)
    {
        PushImage(*image, stream());
        return true;
    }
    return false;
}
MO_REGISTER_CLASS(tcpserver);

bool GStreamerSink::processImpl()
{
    if(!_initialized || gstreamer_pipeline_param.modified())
    {
        cleanup();
        _initialized = create_pipeline(gstreamer_pipeline);
        if(_initialized)
        {
            gstreamer_pipeline_param.modified(false);
        }
    }
    if(_initialized)
    {
        PushImage(*image, stream());
        return true;
    }
    return false;
}
MO_REGISTER_CLASS(GStreamerSink);

JPEGSink::JPEGSink(){
    glib_thread::instance()->start_thread();
    gstreamer_context = mo::Context::create();
    gstreamer_context->thread_id = glib_thread::instance()->get_thread_id();
    //this->_ctx = &gstreamer_context;
}


bool JPEGSink::processImpl(){
    if(gstreamer_pipeline_param.modified()){
        this->cleanup();
        this->create_pipeline(gstreamer_pipeline);
        this->set_caps("image/jpeg");
        this->start_pipeline();
        gstreamer_pipeline_param.modified(false);
    }
    _modified = true;
    return true;
}

GstFlowReturn JPEGSink::on_pull(){
    GstSample *sample = gst_base_sink_get_last_sample(GST_BASE_SINK(_appsink));
    if (sample){
        GstBuffer *buffer;
        GstCaps *caps;
        //GstStructure *s;
        GstMapInfo map;
        caps = gst_sample_get_caps(sample);
        if (!caps){
            LOG(debug) << "could not get sample caps";
            return GST_FLOW_OK;
        }
        buffer = gst_sample_get_buffer(sample);
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)){
            cv::Mat mapped(1, map.size, CV_8U);
            memcpy(mapped.data, map.data, map.size);
            auto ts = mo::getCurrentTime();
            this->jpeg_buffer_param.updateData(mapped, mo::tag::_timestamp = ts, &gstreamer_context);
            if(decoded_param.hasSubscriptions()){
                decoded_param.updateData(cv::imdecode(mapped, cv::IMREAD_UNCHANGED, &decode_buffer),
                    mo::tag::_timestamp = ts, &gstreamer_context);
            }
        }
        gst_sample_unref(sample);
        
    }
    return GST_FLOW_OK;
}

MO_REGISTER_CLASS(JPEGSink)

void BufferedHeartbeatRtsp::nodeInit(bool firstInit)
{

}

