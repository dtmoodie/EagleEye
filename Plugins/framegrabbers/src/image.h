#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include "aqframegrabbers_export.hpp"

namespace aq
{
    namespace nodes
    {
        class aqframegrabbers_EXPORT GrabberImage : public IGrabber
        {
          public:
            static int canLoad(const std::string& path);
            static int loadTimeout();
            MO_DERIVE(GrabberImage, IGrabber)
                ;
                OUTPUT(SyncedMemory, output, {})
            MO_END;
            virtual bool loadData(const std::string& path);
            virtual bool grab();
            cv::Mat image;
        };
    }
}
