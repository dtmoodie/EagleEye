#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include "Aquila/rcc/external_includes/cv_cudacodec.hpp"
#include "Aquila/rcc/external_includes/cv_imgcodec.hpp"
#include "Aquila/rcc/external_includes/cv_videoio.hpp"
#include <MetaObject/params/detail/TParamPtrImpl.hpp>
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

namespace aq
{
    namespace nodes
    {
        class GrabberCV : public IGrabber
        {
        public:
            MO_ABSTRACT(GrabberCV, IGrabber)
                PROPERTY(cv::Ptr<cv::VideoCapture>, h_cam, cv::Ptr<cv::VideoCapture>())
                PROPERTY(cv::Ptr<cv::cudacodec::VideoReader>, d_cam, cv::Ptr<cv::cudacodec::VideoReader>())
                MO_SIGNAL(void, eos)
                SOURCE(SyncedMemory, image, {})
                APPEND_FLAGS(image, mo::ParamFlags::Source_e)
            MO_END;
            bool loadData(const std::string& path);
            bool grab();
        protected:
            virtual bool LoadGPU(const std::string& path);
            virtual bool LoadCPU(const std::string& path);
            mo::OptionalTime_t initial_time;
        };
    }
}
