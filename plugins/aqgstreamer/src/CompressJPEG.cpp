#include "CompressJPEG.hpp"
#include <gst/base/gstbasesink.h>

namespace aqgstreamer
{

CompressImage::~CompressImage()
{
}

bool CompressImage::create_pipeline(const std::string& pipeline_)
{
    if (aq::gstreamer_src_base::create_pipeline(pipeline_))
    {
        _source = (GstAppSrc*)gst_bin_get_by_name(GST_BIN(_pipeline), "mysource");
        if (!_source)
        {
            MO_LOG(warning) << "No appsrc with name \"mysource\" found";
            return false;
        }
        return true;
    }
    return false;
}

CompressImage::CompressImage()
{
    glib_thread::instance()->start_thread();
    m_gstreamer_context = mo::Context::create();
    m_gstreamer_context->thread_id = glib_thread::instance()->get_thread_id();
}

GstFlowReturn CompressImage::on_pull()
{
    GstSample* sample = gst_base_sink_get_last_sample(GST_BASE_SINK(_appsink));
    if (sample)
    {
        GstBuffer* buffer = nullptr;
        GstCaps* caps = nullptr;
        // GstStructure *s;
        GstMapInfo map;
        caps = gst_sample_get_caps(sample);
        if (!caps)
        {
            MO_LOG(debug) << "could not get sample caps";
            return GST_FLOW_OK;
        }
        buffer = gst_sample_get_buffer(sample);
        if (gst_buffer_map(buffer, &map, GST_MAP_READ))
        {
            cv::Mat mapped(1, map.size, CV_8U);
            memcpy(mapped.data, map.data, map.size);
            mo::Time_t ts = mo::ns * GST_BUFFER_DTS(buffer);
            if (GST_BUFFER_DTS(buffer) == 0)
            {
                ts = mo::getCurrentTime();
            }
            output_param.updateData(mapped, mo::tag::_timestamp = ts, m_gstreamer_context.get());
        }
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        gst_buffer_unref(buffer);
    }
    return GST_FLOW_OK;
}

bool CompressImage::processImpl()
{
    if (!_source || !_pipeline)
    {
        this->cleanup();
        std::stringstream ss;
        ss << "appsrc name=mysource ! ";
        if (use_hardware_accel && check_feature("nvvidconv"))
        {
            ss << "nvvidconv ! video/x-raw(memory:NVMM) !";
        }
        else
        {
            ss << "videoconvert ! ";
        }
        if (use_hardware_accel && check_feature("nvjpegenc"))
        {
            ss << "nvjpegenc ! ";
        }
        else
        {
            ss << "jpegenc ! ";
        }
        ss << "appsink name=mysink";
        this->create_pipeline(ss.str());
        gstreamer_src_base::set_caps("image/jpeg");
        auto size = input->getSize();
        gstreamer_sink_base::set_caps(size, 3, 0);
        this->start_pipeline();
    }
    if (_source != nullptr && _pipeline != nullptr)
    {
        pushImage(*input, stream());
        return true;
    }
    return false;
}
}
using namespace aqgstreamer;

MO_REGISTER_CLASS(CompressImage)
