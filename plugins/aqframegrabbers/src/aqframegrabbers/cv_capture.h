#pragma once
#include <aqframegrabbers_export.hpp>

#include <Aquila/framegrabbers/IFrameGrabber.hpp>

#include <Aquila/rcc/external_includes/cv_cudacodec.hpp>
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include <Aquila/rcc/external_includes/cv_videoio.hpp>

#include <Aquila/types/SyncedImage.hpp>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

namespace cv
{
    namespace cudacodec
    {
        class VideoReader;
    }
} // namespace cv

namespace aqframegrabbers
{

    class GrabberCV : public aq::nodes::IGrabber
    {
      public:
        MO_DERIVE(GrabberCV, aq::nodes::IGrabber)
            PARAM(bool, use_system_time, false)
            MO_SIGNAL(void, eos)
            SOURCE(aq::SyncedImage, output)
        MO_END;
        bool loadData(const std::string& path) override;
        bool grab() override;

      protected:
        virtual bool loadGPU(const std::string& path);
        virtual bool loadCPU(const std::string& path);

        mo::OptionalTime initial_time;
        bool query_time = true;
        bool query_frame_number = true;

        cv::Ptr<cv::cudacodec::VideoReader> d_cam;
        cv::Ptr<cv::VideoCapture> h_cam;
    };

} // namespace aqframegrabbers
