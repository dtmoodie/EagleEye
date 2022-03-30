#include "FrameGrabberHTTP.hpp" x
#include <Aquila/framegrabbers/GrabberInfo.hpp>
#include <gst/base/gstbasesink.h>

namespace aqgstreamer
{

    int FrameGrabberHTTP::canLoad(const std::string& doc)
    {
        if (doc.find("http://") != std::string::npos)
        {
            return 10;
        }
        return 0;
    }

    bool FrameGrabberHTTP::loadData(const ::std::string& file_path_)
    {
        std::stringstream ss;
        auto pos = file_path_.find("http://");
        std::string file_path;
        if (pos != std::string::npos)
        {
            file_path = file_path_.substr(pos + 7);
        }
        else
        {
            file_path = file_path_;
        }

        pos = file_path.find(':');
        if (pos == std::string::npos)
        {
            return false;
        }
        getLogger().info("Trying to load {} at port {}", file_path.substr(0, pos), file_path.substr(pos + 1));
        ss << "tcpclientsrc host=" << file_path.substr(0, pos);
        ss << " port=" << file_path.substr(pos + 1);
        ss << " ! matroskademux ! h264parse ! ";
        if (checkFeature("omxh264dec"))
        {
            ss << "omxh264dec ! ";
        }
        else if (checkFeature("avdec_h264"))
        {
            ss << "avdec_h264 ! ";
        }

        if (checkFeature("nvvidconv") && use_nvvidconv)
        {
            ss << "nvvidconv ! ";
        }
        else
        {
            ss << "videoconvert ! ";
        }
        ss << "appsink";
        if (!createPipeline(ss.str()))
        {
            getLogger().warn("Unable to create pipeline for {}", ss.str());
            return false;
        }
        if (!setCaps())
        {
            getLogger().warn("Unable to set caps");
            return false;
        }
        pausePipeline();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
        return startPipeline();
    }

    GstFlowReturn FrameGrabberHTTP::onPull(GstAppSink* appsink)
    {
        GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(appsink));
        if (sample)
        {
            GstBuffer* buffer;
            GstCaps* caps;
            GstStructure* s;
            GstMapInfo map;
            caps = gst_sample_get_caps(sample);
            if (!caps)
            {
                getLogger().debug("could not get sample caps");
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
                getLogger().debug("could not get snapshot dimension");
                return GST_FLOW_OK;
            }
            buffer = gst_sample_get_buffer(sample);
            if (gst_buffer_map(buffer, &map, GST_MAP_READ))
            {
                cv::Mat mapped(height, width, CV_8UC3);
                memcpy(mapped.data, map.data, map.size);
                m_data.enqueue({mapped, buffer->pts});
                sig_update();
            }
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            gst_buffer_unref(buffer);
        }
        return GST_FLOW_OK;
    }

    bool FrameGrabberHTTP::grab()
    {
        Data data;
        bool found = false;
        while (m_data.try_dequeue(data))
        {
            found = true;
        }
        if (found)
        {
            image.publish(data.image, mo::tags::timestamp = mo::Time(std::chrono::nanoseconds(data.pts)));
        }
        return true;
    }
} // namespace aqgstreamer

using namespace aqgstreamer;
MO_REGISTER_CLASS(FrameGrabberHTTP)
