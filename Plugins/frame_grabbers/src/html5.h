#pragma once

#include "gstreamer.h"

namespace EagleLib
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