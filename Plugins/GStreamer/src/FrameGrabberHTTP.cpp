#include "FrameGrabberHTTP.hpp"
#include "Aquila/framegrabbers/FrameGrabberInfo.hpp"
#include <gst/base/gstbasesink.h>
using namespace aq;
using namespace aq::nodes;

int FrameGrabberHTTP::canLoad(const std::string& doc)
{
    if (doc.find("http://") != std::string::npos)
        return 10;
    return 0;
}

bool FrameGrabberHTTP::loadData(const ::std::string& file_path_)
{
    std::stringstream ss;
    auto pos = file_path_.find("http://");
    std::string file_path;
    if (pos != std::string::npos)
        file_path = file_path_.substr(pos + 7);
    else
        file_path = file_path_;

    pos = file_path.find(':');
    if (pos == std::string::npos)
        return false;
    MO_LOG(info) << "Trying to load " << file_path.substr(0, pos) << " at port " << file_path.substr(pos + 1);
    ss << "tcpclientsrc host=" << file_path.substr(0, pos);
    ss << " port=" << file_path.substr(pos + 1);
    ss << " ! matroskademux ! h264parse ! ";
    if (this->check_feature("omxh264dec"))
    {
        ss << "omxh264dec ! ";
    }
    else if (this->check_feature("avdec_h264"))
    {
        ss << "avdec_h264 ! ";
    }
    ss << "videoconvert ! appsink";
    if (!this->create_pipeline(ss.str()))
    {
        MO_LOG(warning) << "Unable to create pipeline for " << ss.str();
        return false;
    }
    if (!this->set_caps())
    {
        MO_LOG(warning) << "Unable to set caps";
        return false;
    }
    this->pause_pipeline();
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return this->start_pipeline();
}

GstFlowReturn FrameGrabberHTTP::on_pull()
{
    GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(_appsink));
    // g_signal_emit_by_name(_appsink, "pull-sample", &sample, NULL);
    if (sample)
    {
        GstBuffer* buffer;
        GstCaps* caps;
        GstStructure* s;
        GstMapInfo map;
        caps = gst_sample_get_caps(sample);
        if (!caps)
        {
            MO_LOG(debug) << "could not get sample caps";
            return GST_FLOW_OK;
        }
        s = gst_caps_get_structure(caps, 0);
        gint width, height;
        gboolean res;
        res = gst_structure_get_int(s, "width", &width);
        res |= gst_structure_get_int(s, "height", &height);
        // const gchar* format = gst_structure_get_string(s, "format");
        if (!res)
        {
            MO_LOG(debug) << "could not get snapshot dimenszion\n";
            return GST_FLOW_OK;
        }
        buffer = gst_sample_get_buffer(sample);
        if (gst_buffer_map(buffer, &map, GST_MAP_READ))
        {
            cv::Mat mapped(height, width, CV_8UC3);
            memcpy(mapped.data, map.data, map.size);
            image_param.updateData(mapped, mo::tag::_timestamp = mo::Time_t(buffer->pts * mo::ns));
        }
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        gst_buffer_unref(buffer);
    }
    return GST_FLOW_OK;
}

MO_REGISTER_CLASS(FrameGrabberHTTP)
