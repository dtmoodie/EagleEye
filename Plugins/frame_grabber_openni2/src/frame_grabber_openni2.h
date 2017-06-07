#pragma once
#include "OpenNI.h"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"

RUNTIME_COMPILER_LINKLIBRARY("OpenNI2.lib");

namespace aq
{
    namespace Nodes
    {
        
        class frame_grabber_openni2: public openni::VideoStream::NewFrameListener, public IFrameGrabber
        {
            openni::VideoFrameRef _frame;
            std::shared_ptr<openni::Device> _device;
            std::shared_ptr<openni::VideoStream> _depth;
        public:
            MO_DERIVE(frame_grabber_openni2, IFrameGrabber)
                SOURCE(SyncedMemory, xyz, {})
                APPEND_FLAGS(xyz, mo::RequestBuffered_e)
            MO_END;
            
            ~frame_grabber_openni2();
        
            bool loadData(std::string file_path);
            void onNewFrame(openni::VideoStream& stream);
            bool processImpl(){return true;}

            static int canLoadPath(const std::string& document);
            static int loadTimeout();
            static std::vector<std::string> listLoadablePaths();
        };
    }
}
