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

        void GstreamerImageGrabber::setStream(const mo::IAsyncStreamPtr_t&)
        {
            nodes::IGrabber::setStream(m_gstreamer_stream);
        }

        void GstreamerImageGrabber::initCustom(bool /*first_init*/)
        {
            std::shared_ptr<aqgstreamer::GLibThread> inst = aqgstreamer::GLibThread::instance();
            m_gstreamer_stream = inst->getStream();
            this->setStream(m_gstreamer_stream);
        }

        GstFlowReturn GstreamerImageGrabber::onPull(GstAppSink* appsink)
        {
            // This is called here since this could be called by a thread created by gstreamer
            mo::initThread();
            mo::IAsyncStream::setCurrent(m_gstreamer_stream);
            GstSample* sample_ = gst_base_sink_get_last_sample(GST_BASE_SINK(appsink));
            std::shared_ptr<GstSample> sample(sample_, &gst_sample_unref);
            // g_signal_emit_by_name(_appsink, "pull-sample", &sample, NULL);
            if (sample)
            {
                GstCaps* caps = gst_sample_get_caps(sample.get());
                if (!caps)
                {
                    this->getLogger().debug("could not get sample caps");
                    return GST_FLOW_OK;
                }
                GstStructure* s = gst_caps_get_structure(caps, 0);
                gint width, height;
                gboolean res;
                res = gst_structure_get_int(s, "width", &width);
                res |= gst_structure_get_int(s, "height", &height);

                if (!res)
                {
                    this->getLogger().debug("Could not get image dimension");
                    return GST_FLOW_OK;
                }

                const gchar* format = gst_structure_get_string(s, "format");
                aq::PixelType pixel_type;
                pixel_type.data_type = aq::DataFlag::kUINT8;
                pixel_type.pixel_format = aq::PixelFormat::kBGR;
                if (format)
                {
                    if (std::string("RGBA") == format)
                    {
                        pixel_type.pixel_format = aq::PixelFormat::kRGBA;
                    }
                    else if (std::string("BGRA") == format)
                    {
                        pixel_type.pixel_format = aq::PixelFormat::kRGBA;
                    }
                    else if (std::string("BGR") == format)
                    {
                        pixel_type.pixel_format = aq::PixelFormat::kBGR;
                    }
                    else if (std::string("RGB") == format)
                    {
                        pixel_type.pixel_format = aq::PixelFormat::kRGB;
                    }
                }

                std::shared_ptr<GstBuffer> buffer = aqgstreamer::ownBuffer(gst_sample_get_buffer(sample.get()));
                aq::SyncedImage image;

                const GstClockTime pts = buffer->pts;
                const bool success = aqgstreamer::mapBuffer(std::move(buffer),
                                                            image,
                                                            aq::Shape<2>(height, width),
                                                            pixel_type,
                                                            GstMapFlags::GST_MAP_READ,
                                                            m_gstreamer_stream);

                if (success)
                {

                    mo::Time time = mo::Time(std::chrono::nanoseconds(pts));
                    if (pts == GST_CLOCK_TIME_NONE || use_system_time)
                    {
                        time = mo::Time::now();
                    }

                    this->getLogger().trace("Received image at {}", time);

                    this->image.publish(std::move(image),
                                        mo::Header(time, mo::FrameNumber(m_sample_counter)),
                                        m_gstreamer_stream.get());
                    ++m_sample_counter;
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
} // namespace aq

using namespace aq::grabbers;
MO_REGISTER_CLASS(GstreamerImageGrabber)
