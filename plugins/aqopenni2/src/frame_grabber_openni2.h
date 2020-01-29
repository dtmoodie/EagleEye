#pragma once
#include "OpenNI.h"
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/types/SyncedMemory.hpp>

RUNTIME_COMPILER_LINKLIBRARY("OpenNI2.lib");

namespace aq
{
    namespace nodes
    {

        class frame_grabber_openni2 : public openni::VideoStream::NewFrameListener, public IFrameGrabber
        {
            openni::VideoFrameRef _frame;
            openni::VideoFrameRef _color_frame;
            std::shared_ptr<openni::Device> _device;
            std::shared_ptr<openni::VideoStream> _depth;
            std::shared_ptr<openni::VideoStream> _color;

          public:
            MO_DERIVE(frame_grabber_openni2, IFrameGrabber)
                SOURCE(SyncedMemory, xyz, {})
                SOURCE(SyncedMemory, depth, {})
                SOURCE(SyncedMemory, color, {})
            MO_END;

            ~frame_grabber_openni2();

            virtual bool loadData(std::string file_path) override;
            virtual void onNewFrame(openni::VideoStream& stream) override;
            virtual bool processImpl() override;

            static int canLoadPath(const std::string& document);
            static int loadTimeout();
            static std::vector<std::string> listLoadablePaths();

            cv::Mat new_xyz;
            cv::Mat new_depth;
            size_t depth_fn;
            cv::Mat new_color;
            size_t color_fn;
        };
    }
}
