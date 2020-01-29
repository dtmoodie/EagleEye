#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/rcc/external_includes/cv_cudacodec.hpp"
#include "Aquila/rcc/external_includes/cv_imgcodec.hpp"
#include "Aquila/rcc/external_includes/cv_videoio.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
#include <MetaObject/params/detail/TParamPtrImpl.hpp>

namespace cv
{
namespace cudacodec
{
class VideoReader;
}
}

namespace aq
{
namespace nodes
{
class GrabberCV : public IGrabber
{
  public:
    MO_ABSTRACT(GrabberCV, IGrabber)
        STATE(cv::Ptr<cv::VideoCapture>, h_cam, cv::Ptr<cv::VideoCapture>())
        STATE(cv::Ptr<cv::cudacodec::VideoReader>, d_cam, cv::Ptr<cv::cudacodec::VideoReader>())
        PARAM(bool, use_system_time, false)
        MO_SIGNAL(void, eos)
        SOURCE(SyncedMemory, image, {})
        APPEND_FLAGS(image, mo::ParamFlags::Source_e)
    MO_END;
    virtual bool loadData(const std::string& path) override;
    virtual bool grab() override;

  protected:
    virtual bool LoadGPU(const std::string& path);
    virtual bool LoadCPU(const std::string& path);
    mo::OptionalTime_t initial_time;
    bool query_time = true;
    bool query_frame_number = true;
};
}
}
