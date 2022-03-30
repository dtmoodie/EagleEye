#include "CompressJPEG.hpp"
#include <gst/base/gstbasesink.h>

namespace aqgstreamer
{

    CompressImage::~CompressImage() {}

    bool CompressImage::createPipeline(const std::string& pipeline_)
    {
        if (GstreamerSrcBase::createPipeline(pipeline_))
        {
            m_source = (GstAppSrc*)gst_bin_get_by_name(GST_BIN(m_pipeline), "mysource");
            if (!m_source)
            {
                getLogger().warn("No appsrc with name \"mysource\" found");
                return false;
            }
            return true;
        }
        return false;
    }

    CompressImage::CompressImage()
    {
        GLibThread::instance()->startThread();
        m_gstreamer_stream = GLibThread::instance()->getStream();
    }

    GstFlowReturn CompressImage::onPull(GstAppSink* appsink)
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

            std::shared_ptr<GstBuffer> buffer = ownBuffer(gst_sample_get_buffer(sample));
            mo::Time ts = mo::ns * GST_BUFFER_DTS(buffer.get());
            if (GST_BUFFER_DTS(buffer.get()) == 0)
            {
                ts = mo::Time::now();
            }

            GstMapInfo map;
            if (gst_buffer_map(buffer.get(), &map, GST_MAP_READ))
            {
                std::shared_ptr<void> owning(nullptr, [map, buffer, sample](void*) mutable {
                    gst_buffer_unmap(buffer.get(), &map);
                    gst_sample_unref(sample);
                });
                ct::TArrayView<uint8_t> view(map.data, map.size);
                ce::shared_ptr<aq::SyncedMemory> wrapped =
                    ce::make_shared<aq::SyncedMemory>(aq::SyncedMemory::wrapHost(view, 1, owning));
                aq::ImageEncoding encoding = ct::fromString<aq::ImageEncoding>(this->encoding.getEnum());
                aq::CompressedImage compressed(std::move(wrapped), encoding);
                output.publish(std::move(compressed), mo::tags::timestamp = ts);
            }
        }
        return GST_FLOW_OK;
    }

    bool CompressImage::processImpl()
    {
        if (!m_source || !m_pipeline)
        {
            this->cleanup();
            std::stringstream ss;
            ss << "appsrc name=mysource ! ";
            if (use_hardware_accel && checkFeature("nvvidconv"))
            {
                ss << "nvvidconv ! video/x-raw(memory:NVMM) !";
            }
            else
            {
                ss << "videoconvert ! ";
            }
            if (use_hardware_accel && checkFeature("nvjpegenc"))
            {
                ss << "nvjpegenc ! ";
            }
            else
            {
                ss << "jpegenc ! ";
            }
            ss << "appsink name=mysink";
            createPipeline(ss.str());
            GstreamerSrcBase::setCaps("image/jpeg");
            const auto size = input->size();
            GstreamerSinkBase::setCaps(cv::Size(size(1), size(0)), 3, 0);
            startPipeline();
        }
        if (m_source != nullptr && m_pipeline != nullptr)
        {
            pushImage(*input, *getStream());
            return true;
        }
        return false;
    }
} // namespace aqgstreamer
using namespace aqgstreamer;

MO_REGISTER_CLASS(CompressImage)
