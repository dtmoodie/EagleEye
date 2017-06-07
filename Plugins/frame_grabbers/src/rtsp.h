#pragma once
#ifdef HAVE_GSTREAMER
#include "gstreamer.h"

namespace aq
{
    namespace Nodes
    {
        
    class frame_grabbers_EXPORT GrabberRTSP: public GrabberGstreamer
    {
    public:
        MO_DERIVE(GrabberRTSP, GrabberGstreamer)
        MO_END;
        bool Load(const std::string& file_path);
        static int canLoad(const std::string& document);
        static int loadTimeout();
    };
    }
}
#endif // HAVE_GSTREAMER
