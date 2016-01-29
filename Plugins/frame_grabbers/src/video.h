#pragma once

#include "cv_capture.h"
#include "EagleLib/ICoordinateManager.h"
namespace EagleLib
{
    class frame_grabber_video : public frame_grabber_cv
    {
    public:

        class frame_grabber_video_info : public FrameGrabberInfo
        {
        public:
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
            virtual bool CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
        };
        
        virtual shared_ptr<ICoordinateManager> GetCoordinateManager();

    protected:
        shared_ptr<ICoordinateManager>          coordinate_manager;
    };
}