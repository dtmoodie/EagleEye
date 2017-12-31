#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "GStreamerExport.hpp"
#include "gstreamer.hpp"
namespace aq
{
    namespace nodes
    {

        class GStreamer_EXPORT FrameGrabberHTTP : virtual public gstreamer_src_base, virtual public IGrabber
        {
          public:
            static int canLoad(const std::string& doc);
            static int loadTimeout() { return 10000; }

            MO_DERIVE(FrameGrabberHTTP, IGrabber)
            OUTPUT(SyncedMemory, image, {})
            MO_END;
            virtual GstFlowReturn on_pull();
            bool loadData(const ::std::string& file_path);
            bool grab() { return true; }
          protected:
        };
    }
}
