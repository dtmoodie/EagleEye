#pragma once

#include "OpenNI.h"
#include <EagleLib/Nodes/IFrameGrabber.hpp>
#include "RuntimeLinkLibrary.h"
SETUP_PROJECT_DEF
RUNTIME_COMPILER_LINKLIBRARY("OpenNI2.lib");

namespace EagleLib
{
    namespace Nodes
    {
        
        class PLUGIN_EXPORTS frame_grabber_openni2: public FrameGrabberBuffered, public openni::VideoStream::NewFrameListener
        {
            openni::VideoFrameRef _frame;
            std::shared_ptr<openni::Device> _device;
            std::shared_ptr<openni::VideoStream> _depth;
        public:
            MO_DERIVE(frame_grabber_openni2, FrameGrabberBuffered)
            MO_END;
            frame_grabber_openni2();
            ~frame_grabber_openni2();
        
            bool LoadFile(const std::string& file_path);
            rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
            long long GetNumFrames();
            void onNewFrame(openni::VideoStream& stream);

            static int CanLoadDocument(const std::string& document);
            static int LoadTimeout();
            static std::vector<std::string> ListLoadableDocuments();
        };
    }
}