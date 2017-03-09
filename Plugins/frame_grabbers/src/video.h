#pragma once

#include "cv_capture.h"
#include "EagleLib/ICoordinateManager.h"
namespace EagleLib
{
    namespace Nodes
    {
    class frame_grabber_video : public frame_grabber_cv
    {
    public:
        ~frame_grabber_video();
        MO_DERIVE(frame_grabber_video, frame_grabber_cv);
        MO_END;
        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
        static int CanLoadDocument(const std::string& document);
    protected:
        rcc::shared_ptr<ICoordinateManager>          coordinate_manager;
    };
    }
}
