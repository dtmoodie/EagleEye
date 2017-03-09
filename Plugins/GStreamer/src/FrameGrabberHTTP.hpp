#include "gstreamer.hpp"
#include "EagleLib/Nodes/IFrameGrabber.hpp"
#include "GStreamerExport.hpp"
namespace EagleLib
{
namespace Nodes
{

class GStreamer_EXPORT FrameGrabberHTTP:
        virtual public gstreamer_src_base,
        virtual public FrameGrabberBuffered
{
public:
    static int CanLoadDocument(const std::string& doc);
    static int LoadTimeout(){return 10000;}

    long long GetNumFrames() {return -1;}
    rcc::shared_ptr<EagleLib::ICoordinateManager> GetCoordinateManager()
    {
        return {};
    }

    MO_DERIVE(FrameGrabberHTTP, FrameGrabberBuffered)

    MO_END;
    virtual GstFlowReturn on_pull();
    bool LoadFile(const ::std::string& file_path);
protected:
};

}
}
