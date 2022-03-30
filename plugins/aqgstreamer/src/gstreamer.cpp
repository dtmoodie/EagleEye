#include <opencv2/core.hpp>

#include "glib_thread.h"
#include "gstreamer.hpp"

#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/core/SystemTable.hpp>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

namespace aqgstreamer
{
    std::shared_ptr<GstBuffer> ownBuffer(GstBuffer* buffer)
    {
        // clang-format off
        return std::shared_ptr<GstBuffer>(buffer, [](GstBuffer* buffer)
        {
            gst_buffer_unref(buffer);
        });
        // clang-format on
    }

    bool mapBuffer(std::shared_ptr<GstBuffer> buffer,
                   std::shared_ptr<cv::Mat>& output,
                   cv::Size size,
                   int32_t type,
                   GstMapFlags flags)
    {
        GstMapInfo map;
        if (gst_buffer_map(buffer.get(), &map, flags))
        {
            cv::Mat wrapping(size, type, map.data);
            auto free_func = [map, buffer](cv::Mat* mat) mutable {
                delete mat;
                gst_buffer_unmap(buffer.get(), &map);
            };

            output = std::shared_ptr<cv::Mat>(new cv::Mat(wrapping), free_func);
            return true;
        }
        return false;
    }

    bool mapBuffer(std::shared_ptr<GstBuffer> buffer, std::shared_ptr<cv::Mat>& output, GstMapFlags flags)
    {
        GstMapInfo map;
        if (gst_buffer_map(buffer.get(), &map, flags))
        {
            cv::Mat wrapping(1, map.size, CV_8U, map.data);
            auto free_func = [map, buffer](cv::Mat* mat) mutable {
                delete mat;
                gst_buffer_unmap(buffer.get(), &map);
            };

            output = std::shared_ptr<cv::Mat>(new cv::Mat(wrapping), free_func);
            return true;
        }
        return false;
    }

    bool mapBuffer(std::shared_ptr<GstBuffer> buffer,
                   aq::SyncedImage& output,
                   aq::Shape<2> size,
                   aq::PixelType type,
                   GstMapFlags flags,
                   mo::IAsyncStreamPtr_t stream)
    {
        std::shared_ptr<GstMapInfo> map(new GstMapInfo, [buffer](GstMapInfo* map) {
            gst_buffer_unmap(buffer.get(), map);
            delete map;
        });
        if (gst_buffer_map(buffer.get(), map.get(), flags))
        {

            if (flags == GstMapFlags::GST_MAP_READ)
            {
                output = aq::SyncedImage(size, type, static_cast<const void*>(map->data), map, stream);
            }
            else
            {
                output = aq::SyncedImage(size, type, static_cast<void*>(map->data), map, stream);
            }
            return true;
        }
        return false;
    }

    bool mapBuffer(std::shared_ptr<GstBuffer> buffer, ce::shared_ptr<aq::SyncedMemory>& output, GstMapFlags flags)
    {
        GstMapInfo map;
        if (gst_buffer_map(buffer.get(), &map, flags))
        {
            std::shared_ptr<GstBuffer> tmp(buffer.get(), [buffer, map](GstBuffer*) mutable {
                gst_buffer_unmap(buffer.get(), &map);
                buffer.reset();
            });

            ct::TArrayView<const void> wrapping(map.data, map.size);
            output = ce::make_shared<aq::SyncedMemory>(aq::SyncedMemory::copyHost(wrapping, 1));
            return true;
        }
        return false;
    }

    // handled messages from the pipeline
    static gboolean busMessage(GstBus* bus, GstMessage* message, void* app)
    {
        (void)bus;
        (void)app;
        MO_LOG(debug, "Received message type: {}", gst_message_type_get_name(GST_MESSAGE_TYPE(message)));

        switch (GST_MESSAGE_TYPE(message))
        {
        case GST_MESSAGE_ERROR: {
            GError* err = NULL;
            gchar* dbg_info = NULL;

            gst_message_parse_error(message, &err, &dbg_info);
            MO_LOG(error, "Error from element {}: {}", GST_OBJECT_NAME(message->src), err->message);
            MO_LOG(error, "Debugging info: {}", dbg_info ? dbg_info : "none");
            g_error_free(err);
            g_free(dbg_info);
            break;
        }
        case GST_MESSAGE_EOS:
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            GstState oldstate, newstate, pendingstate;
            gst_message_parse_state_changed(message, &oldstate, &newstate, &pendingstate);
            switch (newstate)
            {
            case GST_STATE_VOID_PENDING: {
                MO_LOG(debug, "State changed to GST_STATE_VOID_PENDING");
                break;
            }
            case GST_STATE_NULL: {
                MO_LOG(debug, "State changed to GST_STATE_NULL");
                break;
            }
            case GST_STATE_READY: {
                MO_LOG(debug, "State changed to GST_STATE_READY");
                break;
            }
            case GST_STATE_PAUSED: {
                MO_LOG(debug, "State changed to GST_STATE_PAUSED");
                break;
            }
            case GST_STATE_PLAYING: {
                MO_LOG(debug, "State changed to GST_STATE_PLAYING");
                break;
            }
            }
            break;
        }
        default:
            break;
        }
        return TRUE;
    }

    static void _start_feed(GstElement* pipeline, guint size, GstreamerSinkBase* app)
    {
        (void)pipeline;
        (void)size;
        app->startFeed();
    }

    static void _stop_feed(GstElement* pipeline, GstreamerSinkBase* app)
    {
        (void)pipeline;
        app->stopFeed();
    }

    GstreamerBase::GstreamerBase()
    {
        m_pipeline = nullptr;
        if (!gst_is_initialized())
        {
            std::string str("-vvv");
            std::vector<char*> vec({const_cast<char*>(str.c_str())});
            int argc = 1;
            char** argv = vec.data();
            gst_init(&argc, &argv);
            // glib_thread::instance()->start_thread();
        }
    }

    GstreamerBase::~GstreamerBase() { cleanup(); }

    void GstreamerBase::cleanup()
    {

        if (m_pipeline)
        {
            MO_LOG(debug, "Cleaning up pipeline");
            gst_element_set_state(m_pipeline, GST_STATE_NULL);
            gst_object_unref(m_pipeline);
            m_pipeline = nullptr;
        }
    }

    bool GstreamerBase::createPipeline(const std::string& pipeline_)
    {
        cleanup();
        MO_ASSERT(!pipeline_.empty());
        GLibThread::instance()->startThread();

        MO_LOG(info, "Attempting to create pipeline: {}", pipeline_);
        GError* error = nullptr;
        m_pipeline = gst_parse_launch(pipeline_.c_str(), &error);

        if (m_pipeline == nullptr)
        {
            MO_LOG(error, "Error parsing pipeline {}", pipeline_);
            return false;
        }
        else
        {
            MO_LOG(info, "Successfully created pipeline");
        }

        if (error != nullptr)
        {
            MO_LOG(error, "Error parsing pipeline '{}' error = {}", pipeline_, error->message);
            return false;
        }
        MO_LOG(debug, "Input pipeline parsed {}", pipeline_);
        // Error callback
        auto bus = gst_pipeline_get_bus(GST_PIPELINE(m_pipeline));
        if (!bus)
        {
            MO_LOG(error, "Unable to get bus from pipeline");
            return false;
        }
        gst_bus_add_watch(bus, (GstBusFunc)busMessage, this);
        gst_object_unref(bus);
        MO_LOG(debug, "Successfully created pipeline");
        return true;
    }

    bool GstreamerBase::startPipeline()
    {
        if (!m_pipeline)
        {
            return false;
        }
        GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            MO_LOG(error, "Unable to start pipeline");
            return false;
        }
        MO_LOG(debug, "Starting pipeline");
        return true;
    }

    bool GstreamerBase::stopPipeline()
    {
        if (!m_pipeline)
            return false;
        GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_NULL);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            MO_LOG(error, "Unable to stop pipeline");
            return false;
        }
        MO_LOG(debug, "Stopping pipeline");
        return true;
    }

    bool GstreamerBase::pausePipeline()
    {
        if (!m_pipeline)
        {
            return false;
        }
        GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_PAUSED);

        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            MO_LOG(error, "Unable to pause pipeline");
            return false;
        }
        MO_LOG(debug, "Pausing pipeline");
        return true;
    }

    GstState GstreamerBase::getPipelineState()
    {
        if (m_pipeline)
        {
            GstState ret;
            GstState pending;
            if (gst_element_get_state(m_pipeline, &ret, &pending, GST_CLOCK_TIME_NONE) != GST_STATE_CHANGE_FAILURE)
            {
                return ret;
            }
        }
        return GST_STATE_NULL;
    }

    GstreamerSinkBase::GstreamerSinkBase() : GstreamerBase()
    {
        m_pipeline = nullptr;
        m_source = nullptr;
        m_need_data_id = 0;
        m_enough_data_id = 0;
        m_feed_enabled = false;
        m_caps_set = false;

        gst_debug_set_active(1);
    }

    GstreamerSinkBase::~GstreamerSinkBase()
    {

        if (m_source)
        {
            gst_app_src_end_of_stream(m_source);
        }
        cleanup();
    }

    void GstreamerSinkBase::cleanup()
    {
        if (m_pipeline && m_source)
        {
            MO_LOG(debug, "Disconnecting data request signals");
            g_signal_handler_disconnect(m_source, m_need_data_id);
            g_signal_handler_disconnect(m_source, m_enough_data_id);
            gst_object_unref(m_source);
            m_source = nullptr;
        }
        GstreamerBase::cleanup();
    }

    bool GstreamerSinkBase::createPipeline(const std::string& pipeline_)
    {
        if (GstreamerBase::createPipeline(pipeline_))
        {
            m_source = (GstAppSrc*)gst_bin_get_by_name(GST_BIN(m_pipeline), "mysource");
            if (!m_source)
            {
                MO_LOG(warn, "No appsrc with name \"mysource\" found");
                return false;
            }
            return true;
        }
        return false;
    }

    bool GstreamerSinkBase::setCaps(const std::string& caps_)
    {
        if (m_source == nullptr)
        {
            return false;
        }

        GstCaps* caps = gst_caps_new_simple(caps_.c_str(), nullptr, nullptr);

        if (caps == nullptr)
        {
            MO_LOG(error, "Error creating caps for appsrc");
            return false;
        }

        gst_app_src_set_caps(GST_APP_SRC(m_source), caps);

        g_object_set(G_OBJECT(m_source), "stream-type", GST_APP_STREAM_TYPE_STREAM, "format", GST_FORMAT_TIME, NULL);

        MO_LOG(debug, "Connecting need/enough data callbacks");
        m_need_data_id = g_signal_connect(m_source, "need-data", G_CALLBACK(_start_feed), this);
        m_enough_data_id = g_signal_connect(m_source, "enough-data", G_CALLBACK(_stop_feed), this);
        return true;
    }

    void GstreamerSinkBase::pushImage(const aq::SyncedImage& img, mo::IAsyncStream& stream, const mo::Time timestamp)
    {
        // MO_LOG_EVERY_N(debug, 100) << "Pushing image onto pipeline";
        std::chrono::nanoseconds delta = std::chrono::milliseconds(33);

        if (m_prev_time)
        {
            delta = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - *m_prev_time);
        }
        m_prev_time = timestamp;

        MO_LOG(trace, "Estimated frame time: {} ns", delta.count());

        const aq::Shape<3> shape = img.shape();
        const aq::PixelType pixel = img.pixelType();
        const size_t pixel_size = img.pixelSize();
        const cv::Size size(shape(1), shape(0));
        if (!m_caps_set)
        {

            if (setCaps(size, shape(2)))
            {
                m_caps_set = true;
                startPipeline();
            }
        }

        if (m_feed_enabled)
        {
            const gsize buffer_length = pixel_size * shape(0) * shape(1);
            std::shared_ptr<GstBuffer> buffer = ownBuffer(gst_buffer_new_and_alloc(buffer_length));
            std::shared_ptr<cv::Mat> wrapping;
            const bool success = mapBuffer(buffer, wrapping, size, pixel.toCvType());
            if (success)
            {
                return;
            }

            // TODO evaluate if it is better to just copy to the output buffer instead of always
            // using SyncedImage::copyTo which downloads to the internal buffer and then copies
            bool sync = false;
            cv::Mat h_mat = img.getMat(&stream, &sync);

            auto push = [this, wrapping, buffer, h_mat, delta, timestamp](mo::IAsyncStream*) mutable {
                h_mat.copyTo(*wrapping);

                GST_BUFFER_PTS(buffer.get()) = m_timestamp;

                GST_BUFFER_DURATION(buffer.get()) = std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count();

                m_timestamp += GST_BUFFER_DURATION(buffer.get());

                GST_BUFFER_DTS(buffer.get()) = timestamp.time_since_epoch().count();

                GstFlowReturn rw;
                g_signal_emit_by_name(m_source, "push-buffer", buffer.get(), &rw);

                if (rw != GST_FLOW_OK)
                {
                    MO_LOG(error, "Error pushing buffer into appsrc {}", rw);
                }
            };
            if (sync)
            {
                stream.pushWork(std::move(push));
            }
            else
            {
                push(&stream);
            }
        }
    }

    static GstFlowReturn gstreamerSrcBaseNewSample(GstElement* element, GstreamerSrcBase* obj)
    {
        GstAppSink* sink = GST_APP_SINK(element);
        MO_ASSERT(sink != nullptr);
        return obj->onPull(sink);
    }

    GstreamerSrcBase::GstreamerSrcBase()
    {
        m_new_sample_id = 0;
        m_new_preroll_id = 0;
        GLibThread::instance();
    }

    GstreamerSrcBase::~GstreamerSrcBase()
    {
        for (auto appsink : m_appsinks)
        {
            g_signal_handler_disconnect(appsink, m_new_sample_id);
            g_signal_handler_disconnect(appsink, m_new_preroll_id);
            gst_object_unref(appsink);
        }
    }
    bool GstreamerSrcBase::createPipeline(const std::string& pipeline_)
    {
        if (GstreamerBase::createPipeline(pipeline_))
        {
            GValue item = G_VALUE_INIT;
            GstBin* bin = GST_BIN(m_pipeline);
            const GType app_sink_type = GST_TYPE_APP_SINK;

            // GstIterator* appsink_iterator = gst_bin_iterate_all_by_interface(bin, GST_TYPE_APP_SINK);
            GstIterator* appsink_iterator = gst_bin_iterate_elements(bin);

            GstIteratorResult result = gst_iterator_next(appsink_iterator, &item);
            bool done = false;
            while (!done)
            {
                switch (result)
                {
                case GST_ITERATOR_OK: {
                    gpointer object = g_value_get_object(&item);
                    GstElement* element = GST_ELEMENT_CAST(gst_object_ref(object));
                    GstElementFactory* factory = gst_element_get_factory(element);
                    GType element_type = gst_element_factory_get_element_type(factory);
                    if (element_type == app_sink_type)
                    {
                        GstAppSink* appsink = GST_APP_SINK(element);
                        m_appsinks.push_back(appsink);
                        g_object_set(G_OBJECT(appsink), "emit-signals", true, NULL);
                        m_new_sample_id =
                            g_signal_connect(appsink, "new-sample", G_CALLBACK(gstreamerSrcBaseNewSample), this);
                        m_new_preroll_id =
                            g_signal_connect(appsink, "new-preroll", G_CALLBACK(gstreamerSrcBaseNewSample), this);
                    }

                    break;
                }

                default:
                    done = true;
                }
                result = gst_iterator_next(appsink_iterator, &item);
            }

            return true;
        }
        return false;
    }

    bool GstreamerSrcBase::setCaps(const std::string& caps_, int32_t index)
    {
        if (m_appsinks.empty())
        {
            return false;
        }

        GstCaps* caps = gst_caps_new_simple(caps_.c_str(), nullptr, nullptr);

        if (caps == nullptr)
        {
            MO_LOG(error, "Error creating caps \"{}\"", caps_);
            return false;
        }
        if (index == -1)
        {
            for (GstAppSink* appsink : m_appsinks)
            {
                gst_app_sink_set_caps(appsink, caps);
            }
        }
        else
        {
            MO_ASSERT(index < m_appsinks.size());
            gst_app_sink_set_caps(m_appsinks[index], caps);
        }

        return true;
    }

    bool GstreamerSrcBase::setCaps()
    {
        if (m_appsinks.empty())
        {
            return false;
        }

        GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", nullptr);
        if (caps == nullptr)
        {
            // MO_LOG(error, "Error creating caps \"{}\"", caps);
            return false;
        }

        {
            GstCaps* rgba = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", nullptr);
            gst_caps_append(caps, rgba);
        }

        {
            GstCaps* argb = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "ARGB", nullptr);
            gst_caps_append(caps, argb);
        }

        {
            GstCaps* bgra = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGRA", nullptr);
            gst_caps_append(caps, bgra);
        }

        {
            GstCaps* abgr = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "ABGR", nullptr);
            gst_caps_append(caps, abgr);
        }

        {
            GstCaps* rgb = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", nullptr);
            gst_caps_append(caps, rgb);
        }

        {
            GstCaps* rgb = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBx", nullptr);
            gst_caps_append(caps, rgb);
        }

        {
            GstCaps* bgr = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGRx", nullptr);
            gst_caps_append(caps, bgr);
        }

        {
            GstCaps* rgb = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "xRGB", nullptr);
            gst_caps_append(caps, rgb);
        }

        {
            GstCaps* bgr = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "xBGR", nullptr);
            gst_caps_append(caps, bgr);
        }

        /*{
            GstCaps* bgr = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "I420", nullptr);
            gst_caps_append(caps, bgr);
        }

        {
            GstCaps* yuv = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "YUV", nullptr);
            gst_caps_append(caps, yuv);
        }*/
        for (GstAppSink* appsink : m_appsinks)
        {
            gst_app_sink_set_caps(appsink, caps);
            caps = gst_app_sink_get_caps(appsink);
            MO_LOG(debug, "Set appsink caps to {}", gst_caps_to_string(caps));
        }

        return true;
    }

    bool GstreamerSinkBase::setCaps(cv::Size img_size, int channels, int depth)
    {
        if (m_source == nullptr)
        {
            return false;
        }
        MO_ASSERT(depth == CV_8U);
        std::string format;
        if (channels == 3)
        {
            format = "BGR";
        }
        else if (channels == 1)
        {
            format = "GRAY8";
        }
        GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                            "format",
                                            G_TYPE_STRING,
                                            format.c_str(),
                                            "width",
                                            G_TYPE_INT,
                                            img_size.width,
                                            "height",
                                            G_TYPE_INT,
                                            img_size.height,
                                            "framerate",
                                            GST_TYPE_FRACTION,
                                            15,
                                            1,
                                            "pixel-aspect-ratio",
                                            GST_TYPE_FRACTION,
                                            1,
                                            1,
                                            NULL);

        if (caps == nullptr)
        {
            MO_LOG(error, "Error creating caps for appsrc");
            return false;
        }

        gst_app_src_set_caps(GST_APP_SRC(m_source), caps);

        g_object_set(G_OBJECT(m_source), "stream-type", GST_APP_STREAM_TYPE_STREAM, "format", GST_FORMAT_TIME, NULL);
        MO_LOG(debug, "Connecting need/enough data callbacks");
        m_need_data_id = g_signal_connect(m_source, "need-data", G_CALLBACK(_start_feed), this);
        m_enough_data_id = g_signal_connect(m_source, "enough-data", G_CALLBACK(_stop_feed), this);
        m_caps_set = true;
        return true;
    }

    void GstreamerSinkBase::startFeed() { m_feed_enabled = true; }

    void GstreamerSinkBase::stopFeed() { m_feed_enabled = false; }

    std::vector<std::string> GstreamerBase::getInterfaces()
    {
        std::vector<std::string> output;
        // TODO figure out how to do with boost
        /*foreach (auto inter, QNetworkInterface::allInterfaces())
        {
            if (inter.flags().testFlag(QNetworkInterface::IsUp) &&
        !inter.flags().testFlag(QNetworkInterface::IsLoopBack))
            {
                foreach (auto entry, inter.addressEntries())
                {
                    output.push_back(entry.ip().toString().toStdString());
                }
            }
        }*/
        return output;
    }

    std::vector<std::string> GstreamerBase::getGstreamerFeatures(const std::string& filter)
    {
        auto registry = gst_registry_get();
        auto plugins = gst_registry_get_plugin_list(registry);
        std::vector<std::string> plugin_names;
        while (plugins)
        {
            auto features = gst_registry_get_feature_list_by_plugin(
                registry, gst_plugin_get_name(static_cast<GstPlugin*>(plugins->data)));
            GstPluginFeature* feature;
            while (features)
            {
                feature = static_cast<GstPluginFeature*>(features->data);
                std::string name(gst_plugin_feature_get_name(feature));
                if (filter.size())
                {
                    if (name.find(filter) != std::string::npos)
                        plugin_names.push_back(name);
                }
                else
                {
                    plugin_names.push_back(name);
                }
                features = features->next;
            }
            gst_plugin_feature_list_free(features);
            plugins = plugins->next;
        }
        gst_plugin_list_free(plugins);
        return plugin_names;
    }

    bool GstreamerBase::checkFeature(const std::string& feature_name)
    {
        auto registry = gst_registry_get();
        auto feature = gst_registry_lookup_feature(registry, feature_name.c_str());
        if (feature)
        {
            gst_object_unref(feature);
            return true;
        }
        return false;
    }

    bool GstreamerBase::isPipeline(const std::string& string)
    {
        (void)string;
        return true;
    }
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // rtsp server
    // -----------------------------------------------------------------------

    void RTSPServer::setup(std::string pipe_override)
    {
        std::stringstream ss;
        if (pipe_override.size() == 0)
        {
            ss << "appsrc name=mysource ! videoconvert ! ";
            if (checkFeature("omxh264enc"))
            {
                ss << "omxh264enc ! ";
            }
            else
            {
                ss << "openh264enc ! ";
            }
            ss << "rtph264pay config-interval=1 pt=96 ! gdppay ! ";

            switch (server_type.getValue())
            {
            case TCP: {
                ss << "tcpserversink host=";
                break;
            }
            case UDP: {
                ss << "udpsink host=";

                if (host.size())
                {
                    ss << host;
                }
                else
                {
                    MO_LOG(warn, "host not set, setting to localhost");
                    host_param.setValue("127.0.0.1");
                    ss << "127.0.0.1";
                }
                break;
            }
            }
            ss << " port=" << port;

            pipe_override = ss.str();
        }
        else
        {
            gst_pipeline_param.setValue(std::string(pipe_override));
        }
        getLogger().info(pipe_override);
        createPipeline(pipe_override);
    }

    void RTSPserverCallback(int, void* /*user_data*/)
    {
        // static_cast<RTSP_server*>(userData)->pushImage();
    }

    bool RTSPServer::processImpl()
    {
        // Download then push an image
        return false;
    }

} // namespace aqgstreamer

using namespace aqgstreamer;
MO_REGISTER_CLASS(RTSPServer);

void initModule(SystemTable* table)
{
    aqgstreamer::GLibThread::instance(table);
}

// http://cgit.freedesktop.org/gstreamer/gst-rtsp-server/tree/examples/test-appsrc.c
#ifdef HAVE_GST_RTSPSERVER
RTSP_server_new::RTSP_server_new()
{
    loop = nullptr;
    server = nullptr;
    factory = nullptr;
    pipeline = nullptr;
    appsrc = nullptr;
    clientCount = 0;
    connected = false;
    first_run = true;
}
RTSP_server_new::~RTSP_server_new()
{
    NODE_MO_LOG(info) << "Shutting down rtsp server";
    if (pipeline)
    {
        if (gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_SUCCESS)
        {
            NODE_MO_LOG(debug) << "gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_SUCCESS";
        }
    }

    g_main_loop_quit(loop);
    if (factory)
        gst_object_unref(factory);
    if (server)
    {
        g_source_remove(server_id);
        gst_object_unref(server);
    }

    glib_thread.join();
    if (loop)
        g_main_loop_unref(loop);
    if (pipeline)
        gst_object_unref(pipeline);
    if (appsrc)
        gst_object_unref(appsrc);
}
void RTSP_server_new::push_image() {}
void RTSP_server_new::onPipeChange() {}
void RTSP_server_new::glibThread()
{
    MO_LOG(info) << "Starting gmain loop";
    if (!g_main_loop_is_running(loop))
    {
        g_main_loop_run(loop);
    }
    MO_LOG(info) << "[RTSP Server] Gmain loop quitting";
}
void RTSP_server_new::setup(std::string pipeOverride) {}
void rtsp_server_new_need_data_callback(GstElement* appsrc, guint unused, gpointer user_data)
{
    MO_LOG(debug) << __FUNCTION__;
    auto node = static_cast<aq::nodes::RTSP_server_new*>(user_data);
    cv::Mat h_buffer;
    node->notifier.wait_and_pop(h_buffer);
    if (!h_buffer.empty() && node->connected)
    {
        int bufferlength = h_buffer.cols * h_buffer.rows * h_buffer.channels();
        auto buffer = gst_buffer_new_and_alloc(bufferlength);

        GstMapInfo map;
        gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_WRITE);
        memcpy(map.data, h_buffer.data, map.size);
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = node->timestamp;

        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(node->delta, GST_SECOND, 1000);
        node->timestamp += GST_BUFFER_DURATION(buffer);

        GstFlowReturn rw;
        g_signal_emit_by_name(appsrc, "push-buffer", buffer, &rw);

        if (rw != GST_FLOW_OK)
        {
            NODE_MO_LOG(error) << "Error pushing buffer into appsrc " << rw;
        }
        gst_buffer_unref(buffer);
    }
}

static gboolean bus_message_new(GstBus* bus, GstMessage* message, RTSP_server_new* app)
{
    MO_LOG(debug) << "Received message type: " << gst_message_type_get_name(GST_MESSAGE_TYPE(message));

    switch (GST_MESSAGE_TYPE(message))
    {
    case GST_MESSAGE_ERROR: {
        GError* err = NULL;
        gchar* dbg_info = NULL;

        gst_message_parse_error(message, &err, &dbg_info);
        BOOST_MO_LOG(error) << "Error from element " << GST_OBJECT_NAME(message->src) << ": " << err->message;
        BOOST_MO_LOG(error) << "Debugging info: " << (dbg_info) ? dbg_info : "none";
        g_error_free(err);
        g_free(dbg_info);
        g_main_loop_quit(app->loop);
        break;
    }
    case GST_MESSAGE_EOS:
        g_main_loop_quit(app->loop);
        break;
    default:
        break;
    }
    return TRUE;
}
void client_close_handler(GstRTSPClient* client, aq::nodes::RTSP_server_new* node)
{
    node->clientCount--;
    MO_LOG(info) << "[RTSP Server] Client Disconnected " << node->clientCount << " " << client;
    if (node->clientCount == 0)
    {
        BOOST_MO_LOG(info) << "[RTSP Server] Setting pipeline state to GST_STATE_NULL and unreffing old pipeline";
        gst_element_set_state(node->pipeline, GST_STATE_NULL);
        gst_object_unref(node->pipeline);
        gst_object_unref(node->appsrc);
        node->appsrc = nullptr;
        node->pipeline = nullptr;
        node->connected = false;
    }
}
void media_configure(GstRTSPMediaFactory* factory, GstRTSPMedia* media, aq::nodes::RTSP_server_new* node)
{
    BOOST_MO_LOG(debug) << __FUNCTION__;
    if (node->imgSize.area() == 0)
    {
        return;
    }

    // get the element used for providing the streams of the media
    node->pipeline = gst_rtsp_media_get_element(media);

    // get our appsrc, we named it 'mysrc' with the name property
    node->appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(node->pipeline), "mysrc");

    BOOST_MO_LOG(info) << "[RTSP Server] Configuring pipeline " << node->clientCount << " " << node->pipeline << " "
                       << node->appsrc;

    // this instructs appsrc that we will be dealing with timed buffer
    gst_util_set_object_arg(G_OBJECT(node->appsrc), "format", "time");

    // configure the caps of the video
    g_object_set(G_OBJECT(node->appsrc),
                 "caps",
                 gst_caps_new_simple("video/x-raw",
                                     "format",
                                     G_TYPE_STRING,
                                     "BGR",
                                     "width",
                                     G_TYPE_INT,
                                     node->imgSize.width,
                                     "height",
                                     G_TYPE_INT,
                                     node->imgSize.height,
                                     "framerate",
                                     GST_TYPE_FRACTION,
                                     30,
                                     1,
                                     NULL),
                 NULL);

    // install the callback that will be called when a buffer is needed
    g_signal_connect(node->appsrc, "need-data", (GCallback)rtsp_server_new_need_data_callback, node);
    // gst_object_unref(appsrc);
    // gst_object_unref(element);
}
void new_client_handler(GstRTSPServer* server, GstRTSPClient* client, aq::nodes::RTSP_server_new* node)
{
    MO_LOG(debug) << __FUNCTION__;
    node->clientCount++;
    node->connected = true;
    MO_LOG(info) << "New client connected " << node->clientCount << " " << client;
    if (node->first_run)
    {
        g_signal_connect(node->factory, "media-configure", G_CALLBACK(media_configure), node);
    }
    g_signal_connect(client, "closed", G_CALLBACK(client_close_handler), node);
    node->first_run = false;
}

void RTSP_server_new::nodeInit(bool firstInit)
{
    if (firstInit)
    {
        GstRTSPMountPoints* mounts = nullptr;
        gst_debug_set_active(1);
        timestamp = 0;
        prevTime = clock();
        if (!gst_is_initialized())
        {
            NODE_MO_LOG(debug) << "Initializing gstreamer";
            char** argv;
            argv = new char* {"-vvv"};
            int argc = 1;
            gst_init(&argc, &argv);
        }
        if (!loop)
        {
            NODE_MO_LOG(debug) << "Creating glib event loop";
            loop = g_main_loop_new(NULL, FALSE);
        }

        // create a server instance
        if (!server)
        {
            MO_LOG(debug) << "Creating new rtsp server";
            server = gst_rtsp_server_new();
        }

        // get the mount points for this server, every server has a default object
        // that be used to map uri mount points to media factories
        if (!mounts)
        {
            MO_LOG(debug) << "Creating new mount points";
            mounts = gst_rtsp_server_get_mount_points(server);
        }

        // make a media factory for a test stream. The default media factory can use
        // gst-launch syntax to create pipelines.
        // any launch line works as long as it contains elements named pay%d. Each
        // element with pay%d names will be a stream
        if (!factory)
        {
            MO_LOG(debug) << "Creating rtsp media factory";
            factory = gst_rtsp_media_factory_new();
        }
        gst_rtsp_media_factory_set_shared(factory, true);

        gst_rtsp_media_factory_set_launch(
            factory, "( appsrc name=mysrc ! videoconvert ! openh264enc ! rtph264pay name=pay0 pt=96 )");

        // notify when our media is ready, This is called whenever someone asks for
        // the media and a new pipeline with our appsrc is created
        // g_signal_connect(factory, "media-configure", G_CALLBACK(media_configure), this);

        // attach the test factory to the /test url
        gst_rtsp_mount_points_add_factory(mounts, "/test", factory);

        // don't need the ref to the mounts anymore
        g_object_unref(mounts);

        // attach the server to the default maincontext
        server_id = gst_rtsp_server_attach(server, NULL);
        g_signal_connect(server, "client-connected", G_CALLBACK(new_client_handler), this);

        glib_thread = boost::thread(std::bind(&RTSP_server_new::glibThread, this));
    }
}
/*void RTSP_server_download_callback(int status, void* user_data)
{
    auto node = static_cast<RTSP_server_new*>(user_data);
    auto buf = node->hostBuffer.getBack();
    if (buf)
    {
        if (!buf->empty())
        {
            node->notifier.push(buf);
        }
    }
}*/

TS<SyncedMemory> RTSP_server_new::doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream)
{
    auto curTime = clock();
    delta = curTime - prevTime;
    prevTime = curTime;
    cv::Mat h_image = img.getMat(stream);
    imgSize = h_image.size();
    cuda::enqueue_callback_async([h_image, this]() -> void { notifier.push(h_image); }, stream);
    return img;
}
/*cv::cuda::GpuMat RTSP_server_new::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    imgSize = img.size();
    auto curTime = clock();
    delta = curTime - prevTime;
    prevTime = curTime;
    auto buf = hostBuffer.getFront();
    img.download(*buf, stream);
    stream.enqueueHostCallback(RTSP_server_download_callback, this);
    return img;
}*/

static aq::nodes::NodeInfo g_registerer_RTSP_server_new("RTSP_server_new", {"Image", "Sink"});
REGISTERCLASS(RTSP_server_new, &g_registerer_RTSP_server_new);
#endif

/*
References
https://www.youtube.com/watch?v=7Xdry76ek5E
http://www.imgportal.net/home/wp-content/uploads/maris-script1.cpp
http://stackoverflow.com/questions/20219401/how-to-push-opencv-images-into-gstreamer-pipeline-to-stream-it-over-the-tcpserve
http://gstreamer.freedesktop.org/data/doc/gstreamer/head/manual/html/section-data-spoof.html

// GStreamer
#include <gstreamer-0.10/gst/app/gstappsrc.h>
#include <gstreamer-0.10/gst/gst.h>
#include <gstreamer-0.10/gst/gstelement.h>
#include <gstreamer-0.10/gst/gstpipeline.h>
#include <gstreamer-0.10/gst/gstutils.h>

// OpenCV
// #include "cv.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


GMainLoop *glib_MainLoop;
unsigned int heartbeat_Intervall; ///< In Milliseconds
boost::shared_ptr<boost::asio::deadline_timer> heartbeat;

GstElement *source_OpenCV;
guint64 imagecounter;

void
GLib_MainLoop() {
    if( !g_main_loop_is_running( glib_MainLoop ) ) {
        std::cout << "Starting glib_MainLoop..." << std::endl;
        g_main_loop_run( glib_MainLoop );
        std::cout << "Starting glib_MainLoop stopped." << std::endl;
    }
};

/// Creates an Image with a red filled Circle and the current Time displayed in it.
cv::Mat
Create_Image() {
    cv::Size size = cv::Size( 640, 480 );
    cv::Mat image = cv::Mat::zeros( size, CV_8UC3 );
    cv::Point center = cv::Point( size.width / 2, size.height / 2 );
    int thickness = -1;
    int lineType = 8;

    cv::circle( image, center, size.width / 4.0, cv::Scalar( 0, 0, 255 ), thickness, lineType );

    std::stringstream current_Time;
    boost::posix_time::time_facet *facet = new boost::posix_time::time_facet( "%Y.%m.%d %H:%M:%S.%f" );
    current_Time.imbue( std::locale( current_Time.getloc(), facet ) );
    current_Time << boost::posix_time::microsec_clock::universal_time();

    int font = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontsize = 1;
    int font_Thickness = 1;

    int baseline = 0;
    cv::Size textSize_01 = cv::getTextSize( current_Time.str(), font, fontsize, font_Thickness , &baseline );
    baseline += font_Thickness;

    cv::Point textOrg_01( ( image.cols - textSize_01.width ) / 2, ( image.rows + textSize_01.height * 2 ) / 2 );
    cv::Scalar textcolour = cv::Scalar( 0, 255, 0 );
    cv::putText( image, current_Time.str(), textOrg_01, font, fontsize, textcolour , font_Thickness , 1 );

    return image;
}

/// Creates a Graph of the created Pipeline including the contained Elements. The environment variable
"GST_DEBUG_DUMP_DOT_DIR" must be set, e.g to /tmp/ to actually create the Graph.
                                          /// Furthermore GST_DEBUG needs to be activated, e.g. with "GST_DEBUG=3".
                                          /// So "GST_DEBUG=3 GST_DEBUG_DUMP_DOT_DIR=/tmp/" ./Sandbox would work.
                                          /// The .dot file can be converted to a e.g. svg-Graphic with the following
command (Package GraphViz): dot -Tsvg -oPipelineGraph.svg PipelineGraph.dot void Create_PipelineGraph( GstElement
*pipeline ) { bool debug_active = gst_debug_is_active(); gst_debug_set_active( 1 ); GST_DEBUG_BIN_TO_DOT_FILE( GST_BIN(
pipeline ), GST_DEBUG_GRAPH_SHOW_ALL, "PipelineGraph" ); gst_debug_set_active( debug_active );
}

void
Push_new_Image( const boost::system::error_code &error ) {
    if( error != 0 ) {
        std::cout << "Error in Timer: " << error.message() << std::endl;
        return;
    }

    cv::Mat image = Create_Image();

    /// OpenCV handles image in BGR, so to get RGB, Channels R and B needs to be swapped.
    cv::cvtColor( image, image, CV_CVTIMG_SWAP_RB );

    {
        /// How do i get the actual bpp and depth out of the cv::Mat?
        GstCaps *caps = gst_caps_new_simple( "video/x-raw-rgb", "width", G_TYPE_INT, image.cols, "height", G_TYPE_INT,
                                            image.rows, "framerate", GST_TYPE_FRACTION, 0, 1, NULL );
        g_object_set( G_OBJECT( source_OpenCV ), "caps", caps, NULL );
        gst_caps_unref( caps );

        IplImage* img = new IplImage( image );
        uchar *IMG_data = ( uchar* ) img->imageData;

        GstBuffer *buffer;
        {
            int bufferlength = image.cols * image.rows * image.channels();
            buffer = gst_buffer_new_and_alloc( bufferlength );

            /// Copy Data from OpenCV to GStreamer
            memcpy( GST_BUFFER_DATA( buffer ), IMG_data, GST_BUFFER_SIZE( buffer ) );

            GST_BUFFER_DURATION( buffer ) = gst_util_uint64_scale( bufferlength, GST_SECOND, 1 );
        }

        /// Setting the Metadata for the image to be pushed.
        {
            GstCaps *caps_Source = NULL;

            std::stringstream video_caps_text;
            video_caps_text << "video/x-raw-rgb,width=(int)" << image.cols << ",height=(int)" << image.rows <<
                ",framerate=(fraction)0/1";
            caps_Source = gst_caps_from_string( video_caps_text.str().c_str() );

            if( !GST_IS_CAPS( caps_Source ) ) {
                std::cout << "Error creating Caps for OpenCV-Source, exiting...";
                exit( 1 );
            }

            gst_app_src_set_caps( GST_APP_SRC( source_OpenCV ), caps_Source );
            gst_buffer_set_caps( buffer, caps_Source );
            gst_caps_unref( caps_Source );
        }

        /// Setting a continues timestamp
        GST_BUFFER_TIMESTAMP( buffer ) = gst_util_uint64_scale( imagecounter * 20, GST_MSECOND, 1 );
        imagecounter += 1;

        /// Push Buffer into GStreamer-Pipeline
        GstFlowReturn rw;
        rw = gst_app_src_push_buffer( GST_APP_SRC( source_OpenCV ), buffer );

        if( rw != GST_FLOW_OK ) {
            std::cout << "Error push buffer to GStreamer-Pipeline, exiting...";

            exit( 1 );
        } else {
            std::cout << "GST_FLOW_OK " << "imagecounter: " << imagecounter << std::endl;
        }

    }

    /// Renew the Heartbeat-Timer
    heartbeat->expires_from_now( boost::posix_time::milliseconds( heartbeat_Intervall ) );
    heartbeat->async_wait( Push_new_Image );
}

int
main( int argc, char **argv ) {
    std::cout << "Sandbox started." << std::endl;

    /// ####################
    /// Initialise Sandbox
    /// ####################
    boost::shared_ptr<boost::asio::io_service> io_service = boost::make_shared<boost::asio::io_service>();
    boost::shared_ptr<boost::asio::io_service::work> work = boost::make_shared<boost::asio::io_service::work>(
        *io_service );
    boost::shared_ptr<boost::thread_group> threadgroup = boost::make_shared<boost::thread_group>();

    /// io_service Callback for continuously feeding into the pipeline of GStreamer.
    /// I've using to push the Buffer into GStreamer as i come available instead of getting informed about an empty
    pipeline by GStreamer-Signals.
                            heartbeat_Intervall = 1000; ///< In Milliseconds
    heartbeat = boost::make_shared<boost::asio::deadline_timer>( ( *( io_service.get() ) ) );

    std::cout << "Initialise GStreamer..." << std::endl;
    gst_init( &argc, &argv );

    glib_MainLoop = g_main_loop_new( NULL, 0 );

    std::cout << "Start GLib_MainLoop..." << std::endl;
    io_service->post( GLib_MainLoop );

    /// create some Workerthreads
    for( std::size_t i = 0; i < 3; ++i )  {
        threadgroup->create_thread( boost::bind( &boost::asio::io_service::run, &( *io_service ) ) );
    }

    /// ####################
    /// Do the actual Work
    /// ####################
    GstElement *pipeline;
    GstElement *converter_FFMpegColorSpace;
    GstElement *converter_VP8_Encoder;
    GstElement *muxer_WebM;
    GstElement *sink_TCPServer;


    /// create GStreamer Elements

    pipeline = gst_pipeline_new( "OpenCV_to_TCPServer" );

    if( !pipeline ) {
        std::cout << "Error creating Pipeline, exiting...";
        return 1;
    }

    {
        source_OpenCV = gst_element_factory_make( "appsrc", "Source_OpenCV" );

        if( !source_OpenCV ) {
            std::cout << "Error creating OpenCV-Source, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), source_OpenCV );
    }

    {
        converter_FFMpegColorSpace = gst_element_factory_make( "ffmpegcolorspace", "Converter_FFMpegColorSpace" );

        if( !converter_FFMpegColorSpace ) {
            std::cout << "Error creating Converter_FFMpegColorSpace, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), converter_FFMpegColorSpace );
    }

    {
        converter_VP8_Encoder = gst_element_factory_make( "vp8enc", "Converter_VP8_Encoder" );

        if( !converter_VP8_Encoder ) {
            std::cout << "Error creating Converter_VP8_Encoder, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), converter_VP8_Encoder );
    }

    {
        muxer_WebM = gst_element_factory_make( "webmmux", "Muxer_WebM" );

        if( !muxer_WebM ) {
            std::cout << "Error creating Muxer_WebM, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), muxer_WebM );
    }

    {
        sink_TCPServer = gst_element_factory_make( "tcpserversink", "Sink_TCPServer" );

        if( !sink_TCPServer ) {
            std::cout << "Error creating Sink_TCPServer, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), sink_TCPServer );
    }


    /// Link GStreamer Elements

    if( !gst_element_link( source_OpenCV, converter_FFMpegColorSpace ) ) {
        std::cout << "Error linking creating source_OpenCV to converter_FFMpegColorSpace, exiting...";
        return 2;
    }

    if( !gst_element_link( converter_FFMpegColorSpace, converter_VP8_Encoder ) ) {
        std::cout << "Error linking creating converter_FFMpegColorSpace to converter_VP8_Encoder, exiting...";
        return 2;
    }

    if( !gst_element_link( converter_VP8_Encoder, muxer_WebM ) ) {
        std::cout << "Error linking creating converter_VP8_Encoder to muxer_WebM, exiting...";
        return 2;
    }

    if( !gst_element_link( muxer_WebM, sink_TCPServer ) ) {
        std::cout << "Error linking creating muxer_WebM to sink_TCPServer, exiting...";
        return 2;
    }


    /// Set State of the GStreamer Pipeline to Playing
    GstStateChangeReturn ret = gst_element_set_state( pipeline, GST_STATE_PLAYING );

    if( ret == GST_STATE_CHANGE_FAILURE ) {
        std::cout << "Error setting GStreamer-Pipeline to playing.";
        return 2;
    }

    Create_PipelineGraph( pipeline );


    /// Start the Heartbeat, that continously creates new Images
    heartbeat->expires_from_now( boost::posix_time::milliseconds( heartbeat_Intervall ) );
    heartbeat->async_wait( Push_new_Image );

    /// ####################
    /// Shutdown the Sandbox
    /// ####################
    std::cout << "Wait some Seconds before joining all Threads and shutdown the Sandbox..." << std::endl;
    boost::this_thread::sleep( boost::posix_time::seconds( 4 ) );

    std::cout << "Shutdown Sandbox..." << std::endl;
    g_main_loop_quit( glib_MainLoop );
    io_service->stop();

    while( !io_service->stopped() ) {
        boost::this_thread::sleep( boost::posix_time::seconds( 1 ) );
    }

    work.reset();
    threadgroup->join_all();

    g_main_loop_unref( glib_MainLoop );

    threadgroup.reset();
    work.reset();
    io_service.reset();

    std::cout << "Sandbox stopped" << std::endl;
}


*/
