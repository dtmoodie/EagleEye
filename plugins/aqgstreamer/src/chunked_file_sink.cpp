#include "chunked_file_sink.h"
#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <gst/base/gstbasesink.h>

namespace aqgstreamer
{

    int ChunkedFileSink::canLoad(const std::string&)
    {
        return 0; // Currently needs to be manually specified
    }

    int ChunkedFileSink::loadTimeout() { return 3000; }

    GstFlowReturn ChunkedFileSink::onPull(GstAppSink* appsink)
    {
        GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(appsink));
        if (sample)
        {
            GstCaps* caps = gst_sample_get_caps(sample);
            if (!caps)
            {
                getLogger().debug("could not get sample caps");
                return GST_FLOW_OK;
            }
            GstStructure* s = gst_caps_get_structure(caps, 0);
            gint width = 0;
            gint height = 0;
            gboolean res = gst_structure_get_int(s, "width", &width);
            res |= gst_structure_get_int(s, "height", &height);
            // const gchar* format = gst_structure_get_string(s, "format");
            if (!res)
            {
                getLogger().debug("could not get snapshot dimension");
                return GST_FLOW_OK;
            }
            std::shared_ptr<GstBuffer> buffer = ownBuffer(gst_sample_get_buffer(sample));
            std::shared_ptr<cv::Mat> mapped;
            const bool success = mapBuffer(buffer, mapped, cv::Size(width, height), CV_8UC3);
            if (success)
            {
                // not sure what we were supposed to do here?
                // TODO do the things
            }
            gst_sample_unref(sample);
        }
        return GST_FLOW_OK;
    }

    bool ChunkedFileSink::loadData(const std::string& file_path)
    {
        if (GstreamerSrcBase::createPipeline(file_path))
        {
            _filesink = gst_bin_get_by_name(GST_BIN(m_pipeline.get()), "filesink0");
            if (_filesink)
            {
                startPipeline();
                return true;
            }
        }
        return false;
    }

    int JpegKeyframer::canLoad(const std::string& doc)
    {
        if (doc.find("http") != std::string::npos && doc.find("mjpg") != std::string::npos)
        {
            return 10;
        }
        return 0;
    }

    int JpegKeyframer::loadTimeout() { return 10000; }

    bool JpegKeyframer::loadData(const std::string& file_path)
    {
        std::stringstream pipeline;
        pipeline << "souphttpsrc location=" << file_path;
        pipeline << " ! multipartdemux ! appsink name=mysink";

        if (createPipeline(pipeline.str()))
        {
            if (setCaps("image/jpeg"))
            {
                startPipeline();
                return true;
            }
        }
        return false;
    }

    GstFlowReturn JpegKeyframer::onPull(GstAppSink* appsink)
    {
        GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(appsink));
        if (sample)
        {
            GstCaps* caps;
            caps = gst_sample_get_caps(sample);
            if (!caps)
            {
                MO_LOG(debug, "could not get sample caps");
                return GST_FLOW_OK;
            }
            ++keyframe_count;
        }
        return GST_FLOW_OK;
    }

    bool GstreamerSink::processImpl()
    {
        if (pipeline_param.getModified() && !image->empty())
        {
            cleanup();
            if (!this->createPipeline(pipeline))
            {
                getLogger().warn("Unable to create pipeline ", pipeline);
                return false;
            }
            const auto num_channels = image->channels();
            const auto size = image->size();
            const auto depth = aq::toCvDepth(image->pixelType().data_type);
            if (!setCaps(cv::Size(size(1), size(0)), num_channels, depth))
            {
                getLogger().warn("Unable to set caps on pipeline");
                return false;
            }
            if (!startPipeline())
            {
                getLogger().warn("Unable to start pipeline {}", pipeline);
                return false;
            }
            pipeline_param.setModified(false);
        }
        if (m_source && m_feed_enabled)
        {
            auto stream = this->getStream();
            pushImage(*image, *stream);
            return true;
        }
        return false;
    }

} // namespace aqgstreamer

using namespace aqgstreamer;

MO_REGISTER_CLASS(ChunkedFileSink);
MO_REGISTER_CLASS(JpegKeyframer);
MO_REGISTER_CLASS(GstreamerSink)
