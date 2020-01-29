#include "gstreamer_image_grabber.hpp"
#include "glib_thread.h"
#include <Aquila/framegrabbers/GrabberInfo.hpp>
#include <gst/base/gstbasesink.h>

namespace aq
{
namespace grabbers
{
int GstreamerImageGrabber::canLoad(const std::string& doc)
{
    if (doc.find("appsink") != std::string::npos)
        return 11;
    return 0;
}

void GstreamerImageGrabber::initCustom(bool first_init)
{
    image_param.setContext(glib_thread::instance()->getContext().get());
}
GstFlowReturn GstreamerImageGrabber::on_pull()
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

        if (!res)
        {
            MO_LOG(debug) << "could not get snapshot dimension\n";
            return GST_FLOW_OK;
        }
        buffer = gst_sample_get_buffer(sample);
        if (gst_buffer_map(buffer, &map, GST_MAP_READ))
        {
            const gchar* format = gst_structure_get_string(s, "format");
            int pixel_format = CV_8UC3;
            if (format)
            {
                if (std::string("RGBA") == format)
                {
                    pixel_format = CV_8UC4;
                    MO_LOG_FIRST_N(info, 1) << "Received image with RGBA pixel format";
                }
            }
            cv::Mat mapped(height, width, pixel_format);
            memcpy(mapped.data, map.data, map.size);
            auto pts = buffer->pts;
            mo::Time_t time(pts * mo::ns);
            if (pts == GST_CLOCK_TIME_NONE || use_system_time)
            {
                time = mo::getCurrentTime();
            }

            MO_LOG(trace) << "Received image at " << time;
            image_param.updateData(mapped, mo::tag::_timestamp = time, mo::tag::_context = mo::Context::getCurrent());
            sig_update();
        }
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        gst_buffer_unref(buffer);
    }
    return GST_FLOW_OK;
}

bool GstreamerImageGrabber::loadData(const ::std::string& pipeline)
{
    if (!this->create_pipeline(pipeline))
    {
        MO_LOG(warning) << "Unable to create pipeline for " << pipeline;
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

bool GstreamerImageGrabber::grab()
{
    return true;
}
}
}

using namespace aq::grabbers;
MO_REGISTER_CLASS(GstreamerImageGrabber)
