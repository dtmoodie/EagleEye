#pragma once
#ifdef HAVE_GSTREAMER
#include "gstreamer.h"

namespace aq
{
    namespace Nodes
    {
    
    class PLUGIN_EXPORTS frame_grabber_html5 : public frame_grabber_gstreamer
    {
    public:
        frame_grabber_html5();
        virtual bool LoadFile(const std::string& file_path);
        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
        static int CanLoadDocument(const std::string& document);
    };
    }
}
#endif // HAVE_GSTREAMER