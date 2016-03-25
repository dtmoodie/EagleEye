#pragma once

#include "cv_capture.h"
#include "EagleLib/ICoordinateManager.h"
namespace EagleLib
{
    class PLUGIN_EXPORTS frame_grabber_gstreamer: public frame_grabber_cv
    {
    public:
        class PLUGIN_EXPORTS frame_grabber_gstreamer_info: public FrameGrabberInfo
        {
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
            virtual int CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
        };
        frame_grabber_gstreamer();
        virtual bool LoadFile(const std::string& file_path);
        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
    protected:

        cv::Ptr<cv::VideoCapture> h_cam;
        rcc::shared_ptr<ICoordinateManager>          coordinate_manager;
        std::string                             loaded_file;
        TS<SyncedMemory>                        current_frame;
    };
}