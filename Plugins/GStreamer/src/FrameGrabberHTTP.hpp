#include "gstreamer.hpp"
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "GStreamerExport.hpp"
namespace aq
{
namespace Nodes
{

class GStreamer_EXPORT FrameGrabberHTTP:
        virtual public gstreamer_src_base,
        virtual public IGrabber
{
public:
    static int CanLoad(const std::string& doc);
    static int Timeout(){return 10000;}

    MO_DERIVE(FrameGrabberHTTP, IGrabber)
        OUTPUT(SyncedMemory, image, {})
    MO_END;
    virtual GstFlowReturn on_pull();
    bool Load(const ::std::string& file_path);
    bool Grab(){return true;}
protected:
};

}
}
