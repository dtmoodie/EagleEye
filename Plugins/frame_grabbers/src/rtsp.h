#pragma once

#include "gstreamer.h"
#include "EagleLib/ICoordinateManager.h"

namespace EagleLib
{
    class frame_grabber_rtsp: public frame_grabber_gstreamer
    {
    public:
        class frame_grabber_rtsp_info: public FrameGrabberInfo
        {
        public:
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
            virtual bool CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
            virtual int LoadTimeout() const;
        };
        frame_grabber_rtsp();
        virtual void Init(bool firstInit);
        virtual bool LoadFile(const std::string& file_path = "");
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);

        virtual shared_ptr<ICoordinateManager> GetCoordinateManager();
        
    protected:
        shared_ptr<ICoordinateManager>          coordinate_manager;
        size_t frame_count;
        bool _reconnect;
    };
}