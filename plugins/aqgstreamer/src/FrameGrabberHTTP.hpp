#ifndef AQGSTREAMER_FRAME_GRABBER_HTTP_HPP
#define AQGSTREAMER_FRAME_GRABBER_HTTP_HPP
#include "gstreamer.hpp"

#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "aqgstreamer/aqgstreamer_export.hpp"

namespace aqgstreamer
{

    class aqgstreamer_EXPORT FrameGrabberHTTP : virtual public GstreamerSrcBase, virtual public aq::nodes::IGrabber
    {
      public:
        static int canLoad(const std::string& doc);
        static int loadTimeout() { return 10000; }

        MO_DERIVE(FrameGrabberHTTP, IGrabber)
            SOURCE(aq::SyncedImage, image)
            PARAM(bool, use_nvvidconv, false)
            MO_SIGNAL(void, update)
        MO_END;

        GstFlowReturn onPull(GstAppSink*) override;
        bool loadData(const ::std::string& file_path) override;
        bool grab() override;

      protected:
        struct Data
        {
            aq::SyncedImage image;
            size_t pts;
        };
        moodycamel::ConcurrentQueue<Data> m_data;
    };
} // namespace aqgstreamer

#endif // AQGSTREAMER_FRAME_GRABBER_HTTP_HPP
