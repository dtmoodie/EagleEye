#pragma once
#include "Aquila/Nodes/IFrameGrabber.hpp"
#include "Aquila/rcc/external_includes/cv_cudacodec.hpp"
#include "Aquila/rcc/external_includes/cv_imgcodec.hpp"
#include "Aquila/rcc/external_includes/cv_videoio.hpp"

#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

namespace aq
{
    namespace Nodes
    {
        class GrabberCV : public IGrabber
        {
        public:
            MO_ABSTRACT(GrabberCV, IGrabber)
                PROPERTY(cv::Ptr<cv::VideoCapture>, h_cam, cv::Ptr<cv::VideoCapture>())
                PROPERTY(cv::Ptr<cv::cudacodec::VideoReader>, d_cam, cv::Ptr<cv::cudacodec::VideoReader>())
                MO_SIGNAL(void, eos)
                OUTPUT(SyncedMemory, image, {})
                APPEND_FLAGS(image, mo::Source_e)
            MO_END;
            bool Load(const std::string& path);
            bool Grab();
        protected:
            virtual bool LoadGPU(const std::string& path);
            virtual bool LoadCPU(const std::string& path);
            boost::posix_time::ptime initial_time;
        };
    }
}
