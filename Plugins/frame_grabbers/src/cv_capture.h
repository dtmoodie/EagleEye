#pragma once

#include "EagleLib/frame_grabber_base.h"
#include "EagleLib/ICoordinateManager.h"
#include "EagleLib/rcc/external_includes/cv_cudacodec.hpp"
#include "EagleLib/rcc/external_includes/cv_imgcodec.hpp"
#include "EagleLib/rcc/external_includes/cv_videoio.hpp"

#include "RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("cv_capture", ".cpp");

namespace EagleLib
{
    class frame_grabber_cv: public FrameGrabberBuffered
    {
    public:
        frame_grabber_cv();
        virtual bool LoadFile(const std::string& file_path);
        virtual bool d_LoadFile(const std::string& file_path);
        virtual bool h_LoadFile(const std::string& file_path);
        virtual int GetNumFrames();
        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(bool firstInit);

    protected:
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);

        cv::Ptr<cv::VideoCapture>               h_cam;
        cv::Ptr<cv::cudacodec::VideoReader>     d_cam;
        TS<SyncedMemory>                        current_frame;

    };
}