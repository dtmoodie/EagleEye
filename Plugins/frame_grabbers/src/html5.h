#pragma once
#ifdef HAVE_GSTREAMER
#include "gstreamer.h"
#include "frame_grabbersExport.hpp"
namespace aq
{
namespace Nodes
{
    
    class frame_grabbers_EXPORT GrabberHTML : public GrabberGstreamer
    {
    public:
        MO_DERIVE(GrabberHTML, GrabberGstreamer)

        MO_END;
        virtual bool Load(const std::string& file_path);
        static int canLoad(const std::string& document);
    };
}
}
#endif // HAVE_GSTREAMER
