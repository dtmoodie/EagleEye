#pragma once

#include "cv_capture.h"
#include "EagleLib/ICoordinateManager.h"
namespace EagleLib
{
    class PLUGIN_EXPORTS frame_grabber_video : public frame_grabber_cv
    {
    public:

        class PLUGIN_EXPORTS frame_grabber_video_info : public FrameGrabberInfo
        {
        public:
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
            virtual int CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
        };
        
        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();

    protected:
        rcc::shared_ptr<ICoordinateManager>          coordinate_manager;
    };
}