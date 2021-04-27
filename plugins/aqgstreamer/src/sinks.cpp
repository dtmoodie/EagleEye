#include "sinks.hpp"
#include "glib_thread.h"
#include <Aquila/nodes/NodeImpl.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include <gst/base/gstbasesink.h>
#include <gst/gst.h>
#include <opencv2/imgcodecs.hpp>
using namespace aq;
using namespace aq::nodes;
namespace aqgstreamer
{

    void TCPServer::nodeInit(bool first_init)
    {
        if (first_init)
        {
            encoders.addEnum(-1, "Select encoder");

            if (checkFeature("matroskamux"))
            {
                if (checkFeature("openh264enc"))
                {
                    encoders.addEnum(0, "openh264enc");
                }
                if (checkFeature("avenc_h264"))
                {
                    encoders.addEnum(1, "avenc_h264");
                }
                if (checkFeature("omxh264enc"))
                {
                    encoders.addEnum(2, "omxh264enc");
                }
            }
            if (checkFeature("webmmux"))
            {
                if (checkFeature("omxvp8enc"))
                {
                    encoders.addEnum(3, "omxvp8enc");
                }
                if (checkFeature("vp8enc"))
                {
                    encoders.addEnum(4, "vp8enc");
                }
            }
            encoders_param.setValue(std::move(encoders));
        }
    }

    bool TCPServer::processImpl()
    {
        if (!_initialized || encoders_param.getModified() || interfaces_param.getModified())
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
                _initialized = createPipeline(ss.str());
                if (_initialized)
                {
                    encoders_param.setModified(false);
                    interfaces_param.setModified(false);
                }
            }
        }
        if (_initialized)
        {
            pushImage(*image, mo::IAsyncStream::currentRef());
            return true;
        }
        return false;
    }

    JPEGSink::JPEGSink()
    {
        GLibThread::instance()->startThread();
        m_gstreamer_stream = GLibThread::instance()->getStream();
    }

    bool JPEGSink::processImpl()
    {
        if (pipeline_param.getModified())
        {
            cleanup();
            createPipeline(pipeline);
            setCaps("image/jpeg");
            startPipeline();
            pipeline_param.setModified(false);
        }
        setModified(true);
        return true;
    }

    GstFlowReturn JPEGSink::onPull()
    {
        GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(m_appsink));
        if (sample)
        {

            GstCaps* caps = gst_sample_get_caps(sample);
            if (!caps)
            {
                getLogger().debug("could not get sample caps");
                return GST_FLOW_OK;
            }

            {
                std::shared_ptr<GstBuffer> buffer = ownBuffer(gst_sample_get_buffer(sample));
                ce::shared_ptr<aq::SyncedMemory> memory;
                const bool success = mapBuffer(buffer, memory);
                if (success)
                {
                    const mo::Time ts(buffer->pts);
                    aq::CompressedImage output(std::move(memory), aq::ImageEncoding::JPEG);

                    jpeg_buffer.publish(std::move(output), mo::tags::timestamp = ts, *m_gstreamer_stream);

                    if (decoded.getNumSubscribers())
                    {
                        ct::TArrayView<const uint8_t> view = memory->hostAs<uint8_t>(m_gstreamer_stream.get());
                        cv::Mat tmp(1, view.size(), CV_8U, const_cast<uint8_t*>(view.data()));
                        cv::Mat decoded = cv::imdecode(tmp, cv::IMREAD_UNCHANGED, &m_decode_buffer);
                        this->decoded.publish(std::move(decoded), mo::tags::timestamp = ts, *m_gstreamer_stream);
                    }
                }
            }

            gst_sample_unref(sample);
        }
        return GST_FLOW_OK;
    }

} // namespace aqgstreamer
using namespace aqgstreamer;
MO_REGISTER_CLASS(TCPServer);
MO_REGISTER_CLASS(JPEGSink)
