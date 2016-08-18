#pragma once

#include "EagleLib/frame_grabber_base.h"
#include "gstreamer.hpp"

namespace EagleLib
{
    class PLUGIN_EXPORTS chunked_file_sink: public gstreamer_src_base, public FrameGrabberBuffered
    {
    protected:
        GstElement* _filesink;
    public:
        class PLUGIN_EXPORTS chunked_file_sink_info: public FrameGrabberInfo
        {
            virtual int CanLoadDocument(const std::string& document) const;
            virtual std::string GetObjectName();
            virtual int LoadTimeout() const;
        };
        virtual bool LoadFile(const std::string& file_path);
        virtual int GetNumFrames();
        virtual rcc::shared_ptr<EagleLib::ICoordinateManager> GetCoordinateManager();
        virtual void Init(bool firstInit);
        virtual GstFlowReturn on_pull();
    };
}