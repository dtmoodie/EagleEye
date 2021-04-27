#ifndef AQGSTREAMER_GSTREAMER_HPP
#define AQGSTREAMER_GSTREAMER_HPP

#ifdef HAVE_GST_RTSPSERVER
#include <gst/rtsp-server/rtsp-server.h>
#endif
#include "aqgstreamer/Aquila/rcc/external_includes/aqgstreamer_link_libs.hpp"
#include "aqgstreamer/aqgstreamer_export.hpp"

#include "glib_thread.h"

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/core/detail/Export.hpp>
#include <Aquila/nodes/Node.hpp>

#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/object/MetaObject.hpp>

#include <opencv2/core/types.hpp>

#include <boost/thread.hpp>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstpipeline.h>
#include <gst/gstutils.h>

namespace aqgstreamer
{
    std::shared_ptr<GstBuffer> ownBuffer(GstBuffer*);

    bool mapBuffer(std::shared_ptr<GstBuffer> buffer,
                   std::shared_ptr<cv::Mat>& output,
                   cv::Size size,
                   int32_t type = CV_8UC1,
                   GstMapFlags flags = GstMapFlags::GST_MAP_READ);
    bool mapBuffer(std::shared_ptr<GstBuffer> buffer,
                   std::shared_ptr<cv::Mat>& output,
                   GstMapFlags flags = GstMapFlags::GST_MAP_READ);
    bool mapBuffer(std::shared_ptr<GstBuffer> buffer,
                   ce::shared_ptr<aq::SyncedMemory>& output,
                   GstMapFlags flags = GstMapFlags::GST_MAP_READ);

    class aqgstreamer_EXPORT GstreamerBase
    {
      public:
        GstreamerBase();
        virtual ~GstreamerBase();

        virtual bool createPipeline(const std::string& pipeline_);
        virtual bool startPipeline();
        virtual bool stopPipeline();
        virtual bool pausePipeline();
        virtual GstState getPipelineState();

        static std::vector<std::string> getInterfaces();
        static std::vector<std::string> getGstreamerFeatures(const std::string& filter = "");
        static bool checkFeature(const std::string& feature_name);
        // Attempt to detect if a string is a valid gstreamer pipeline
        static bool isPipeline(const std::string& string);

      protected:
        // The gstreamer pipeline
        GstElement* m_pipeline;
        GstClockTime m_timestamp;
        mo::OptionalTime m_prev_time;
        virtual void cleanup();
        bool m_caps_set = false;
    };

    // used to feed data into EagleEye from gstreamer, use when creating frame grabbers
    class aqgstreamer_EXPORT GstreamerSrcBase : virtual public GstreamerBase
    {
      public:
        GstreamerSrcBase();
        virtual ~GstreamerSrcBase();
        virtual bool createPipeline(const std::string& pipeline_);
        // Called when data is ready to be pulled from the appsink
        virtual GstFlowReturn onPull() = 0;
        virtual bool setCaps(const std::string& caps);
        virtual bool setCaps();

      protected:
        GstAppSink* m_appsink;
        guint m_new_sample_id;
        guint m_new_preroll_id;
    };

    // Used to feed a gstreamer pipeline from EagleEye
    class aqgstreamer_EXPORT GstreamerSinkBase : virtual public GstreamerBase, public aq::nodes::Node
    {
      public:
        MO_DERIVE(GstreamerSinkBase, aq::nodes::Node)

        MO_END;

        GstreamerSinkBase();
        virtual ~GstreamerSinkBase();

        virtual bool createPipeline(const std::string& pipeline_);
        virtual bool setCaps(cv::Size image_size, int channels, int depth = CV_8U);
        virtual bool setCaps(const std::string& caps);

        void
        pushImage(const aq::SyncedImage& img, mo::IAsyncStream& stream, const mo::Time timestamp = mo::Time::now());

        // Used for gstreamer to indicate that the appsrc needs to either feed data or stop feeding data
        virtual void startFeed();
        virtual void stopFeed();

      protected:
        GstAppSrc* m_source;    // The output of Aquila's processing pipeline and the input to the gstreamer pipeline
        guint m_need_data_id;   // id for the need data signal
        guint m_enough_data_id; // id for the enough data signal

        bool m_feed_enabled;
        virtual void cleanup();
    };

    // Decpricated ?
    class aqgstreamer_EXPORT RTSPServer : public GstreamerSinkBase
    {
      public:
        enum ServerType
        {
            TCP = 0,
            UDP = 1
        };

        MO_DERIVE(RTSPServer, aq::nodes::Node)
            INPUT(aq::SyncedImage, image)

            ENUM_PARAM(server_type, TCP, UDP);

            PARAM(unsigned short, port, 8004);
            PARAM(std::string, host, "");
            PARAM(std::string, gst_pipeline, "");

            STATE(time_t, delta, 0);
            STATE(time_t, prevTime, 0);
            STATE(GstClockTime, timestamp, 0);
        MO_END;

        void setup(std::string pipe_override = std::string());

      private:
        bool processImpl() override;
    };

#ifdef HAVE_GST_RTSPSERVER
class aqgstreamer_EXPORT RTSP_server_new : public Node
{
  public:
    GstClockTime timestamp;
    time_t prevTime;
    time_t delta;
    GMainLoop* loop;
    GstRTSPServer* server;
    int clientCount;
    bool connected;
    bool first_run;
    guint server_id;
    GstRTSPMediaFactory* factory;
    GstElement *pipeline, *appsrc;

    boost::thread glib_thread;
    void glibThread();
    concurrent_notifier<cv::Mat> notifier;
    cv::cuda::HostMem* currentNewestFrame;
    RTSP_server_new();
    void push_image();
    void onPipeChange();
    void setup(std::string pipeOverride = std::string());
    ~RTSP_server_new();
    virtual void nodeInit(bool firstInit);
    virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
    cv::Size imgSize;
};
#endif

} // namespace aqgstreamer

extern "C" void initModule(SystemTable* table);
#endif // AQGSTREAMER_GSTREAMER_HPP
