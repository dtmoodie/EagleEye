#pragma once

#include "OpenNI.h"
#include <opencv2/core.hpp>

#include <Aquila/types/SyncedImage.hpp>

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

#include <aqopenni2_export.hpp>

namespace aqopenni2
{
    class frame_grabber_openni2 : public openni::VideoStream::NewFrameListener, public aq::nodes::IFrameGrabber
    {
        openni::VideoFrameRef _frame;
        openni::VideoFrameRef _color_frame;
        std::shared_ptr<openni::Device> _device;
        std::shared_ptr<openni::VideoStream> _depth;
        std::shared_ptr<openni::VideoStream> _color;

      public:
        MO_DERIVE(frame_grabber_openni2, aq::nodes::IFrameGrabber)
            SOURCE(aq::SyncedImage, xyz)
            SOURCE(aq::SyncedImage, depth)
            SOURCE(aq::SyncedImage, color)
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
} // namespace aqopenni2
