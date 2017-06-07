#pragma once
#include "Aquila/types/SyncedMemory.hpp"
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "frame_grabbersExport.hpp"
namespace aq{
    namespace Nodes{
    class frame_grabbers_EXPORT GrabberImage: public IGrabber
    {
    public:
        static int canLoad(const std::string& path);
        static int loadTimeout();
        MO_DERIVE(GrabberImage, IGrabber);
            OUTPUT(SyncedMemory, output, {})
        MO_END;
        virtual bool loadData(const std::string& path);
        virtual bool grab();
        cv::Mat image;
        
    };
    }
}