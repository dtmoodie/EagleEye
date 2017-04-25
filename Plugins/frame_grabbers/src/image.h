#pragma once

#include "Aquila/Nodes/IFrameGrabber.hpp"
#include "frame_grabbersExport.hpp"
namespace aq
{
    namespace Nodes
    {
    
    class frame_grabbers_EXPORT GrabberImage: public IGrabber
    {
    public:
        static int CanLoad(const std::string& path);
        static int Timeout();
        MO_DERIVE(GrabberImage, IGrabber);
            OUTPUT(SyncedMemory, output, {})
        MO_END;
        virtual bool Load(const std::string& path);
        virtual bool Grab();
        cv::Mat image;
        
    };
    }
}