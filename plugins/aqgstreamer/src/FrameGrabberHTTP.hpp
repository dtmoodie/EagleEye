#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "aqgstreamer/aqgstreamer_export.hpp"
#include "gstreamer.hpp"
namespace aq
{
namespace nodes
{

class aqgstreamer_EXPORT FrameGrabberHTTP : virtual public gstreamer_src_base, virtual public IGrabber
{
  public:
    static int canLoad(const std::string& doc);
    static int loadTimeout() { return 10000; }

    MO_DERIVE(FrameGrabberHTTP, IGrabber)
        SOURCE(SyncedMemory, image, {})
        PARAM(bool, use_nvvidconv, false)
        MO_SIGNAL(void, update)
    MO_END
    virtual GstFlowReturn on_pull() override;
    virtual bool loadData(const ::std::string& file_path) override;
    virtual bool grab() override;

  protected:
    struct Data
    {
        SyncedMemory image;
        size_t pts;
    };
    moodycamel::ConcurrentQueue<Data> m_data;
};
}
}
