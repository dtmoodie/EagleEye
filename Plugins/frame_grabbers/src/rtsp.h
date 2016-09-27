#pragma once
#ifdef HAVE_GSTREAMER
#include "gstreamer.h"
#include "EagleLib/ICoordinateManager.h"

namespace EagleLib
{
    namespace Nodes
    {
    
    class PLUGIN_EXPORTS frame_grabber_rtsp: public frame_grabber_gstreamer
    {
    public:
        
        frame_grabber_rtsp();
        virtual void NodeInit(bool firstInit);
        virtual bool LoadFile(const std::string& file_path = "");
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);

        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();

        MO_DERIVE(frame_grabber_rtsp, frame_grabber_gstreamer)
            MO_SLOT(void, seek_relative_msec, double);
        MO_END;

        static int CanLoadDocument(const std::string& document);
        static int LoadTimeout();
    protected:
        rcc::shared_ptr<ICoordinateManager>          coordinate_manager;
        size_t frame_count;
        bool _reconnect;
    };
    }
}
#endif // HAVE_GSTREAMER