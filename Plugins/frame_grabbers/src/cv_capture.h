#pragma once

#include "EagleLib/Nodes/IFrameGrabber.hpp"
#include "EagleLib/ICoordinateManager.h"
#include "EagleLib/rcc/external_includes/cv_cudacodec.hpp"
#include "EagleLib/rcc/external_includes/cv_imgcodec.hpp"
#include "EagleLib/rcc/external_includes/cv_videoio.hpp"

#include "RuntimeSourceDependency.h"



RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("cv_capture", ".cpp");

namespace EagleLib
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS frame_grabber_cv_info : public FrameGrabberInfo
        {
        public:
            virtual std::string GetObjectName();
            virtual int CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
        };
        class PLUGIN_EXPORTS frame_grabber_cv: public FrameGrabberThreaded
        {
        public:
            frame_grabber_cv();
            virtual bool LoadFile(const std::string& file_path);
            virtual bool d_LoadFile(const std::string& file_path);
            virtual bool h_LoadFile(const std::string& file_path);
            
            TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
            long long GetNumFrames();
            void Serialize(ISimpleSerializer* pSerializer);

        protected:
            virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
            virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);

            cv::Ptr<cv::VideoCapture>               h_cam;
            cv::Ptr<cv::cudacodec::VideoReader>     d_cam;
            TS<SyncedMemory>                        current_frame;

        };
    }
}