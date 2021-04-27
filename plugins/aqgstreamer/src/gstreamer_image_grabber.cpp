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
        {
            return 11;
        }
        return 0;
    }

    int GstreamerImageGrabber::loadTimeout() { return 10000; }

    void GstreamerImageGrabber::initCustom(bool /*first_init*/)
    {
        std::shared_ptr<aqgstreamer::GLibThread> inst = aqgstreamer::GLibThread::instance();
        // std::shared_ptr<mo::IAsyncStream> stream = inst->getStream();
        // image.setStream(*stream);
    }

    GstFlowReturn GstreamerImageGrabber::onPull()
    {
        GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(m_appsink));
        // g_signal_emit_by_name(_appsink, "pull-sample", &sample, NULL);
        if (sample)
        {
            GstCaps* caps = gst_sample_get_caps(sample);
            if (!caps)
            {
                getLogger().debug("could not get sample caps");
                return GST_FLOW_OK;
            }
            GstStructure* s = gst_caps_get_structure(caps, 0);
            gint width, height;
            gboolean res;
            res = gst_structure_get_int(s, "width", &width);
            res |= gst_structure_get_int(s, "height", &height);

            if (!res)
            {
                getLogger().debug("Could not get image dimension");
                return GST_FLOW_OK;
            }

            const gchar* format = gst_structure_get_string(s, "format");
            int32_t pixel_format = CV_8UC3;
            if (format)
            {
                if (std::string("RGBA") == format)
                {
                    pixel_format = CV_8UC4;
                }
            }

            std::shared_ptr<GstBuffer> buffer = aqgstreamer::ownBuffer(gst_sample_get_buffer(sample));
            std::shared_ptr<cv::Mat> wrapping;

            const bool success = aqgstreamer::mapBuffer(buffer, wrapping, cv::Size(width, height), pixel_format);

            if (success)
            {
                const GstClockTime pts = buffer->pts;
                mo::Time time = mo::Time(std::chrono::nanoseconds(pts));
                if (pts == GST_CLOCK_TIME_NONE || use_system_time)
                {
                    time = mo::Time::now();
                }

                this->getLogger().trace("Received image at {}", time);
                auto stream = mo::IAsyncStream::current();
                aq::SyncedImage image(*wrapping, aq::PixelFormat::kBGR, stream);
                image.setOwning(wrapping);
                this->image.publish(image, mo::tags::timestamp = time, mo::tags::stream = stream.get());
                sig_update();
            }
        }
        return GST_FLOW_OK;
    }

    bool GstreamerImageGrabber::loadData(const ::std::string& pipeline)
    {
        if (!createPipeline(pipeline))
        {
            getLogger().warn("Unable to create pipeline for {}", pipeline);
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

    bool GstreamerImageGrabber::grab() { return true; }
} // namespace grabbers
}

using namespace aq::grabbers;
MO_REGISTER_CLASS(GstreamerImageGrabber)
