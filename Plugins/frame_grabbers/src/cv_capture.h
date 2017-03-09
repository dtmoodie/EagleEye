#pragma once

#include "Aquila/Nodes/IFrameGrabber.hpp"
#include "Aquila/ICoordinateManager.h"
#include "Aquila/rcc/external_includes/cv_cudacodec.hpp"
#include "Aquila/rcc/external_includes/cv_imgcodec.hpp"
#include "Aquila/rcc/external_includes/cv_videoio.hpp"

#include "RuntimeSourceDependency.h"



RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("cv_capture", ".cpp");

namespace aq
{
    namespace Nodes
    {
        class frame_grabber_cv: public FrameGrabberThreaded
        {
        public:
            frame_grabber_cv();
            MO_ABSTRACT(frame_grabber_cv, FrameGrabberThreaded)
                PROPERTY(cv::Ptr<cv::VideoCapture>, h_cam, cv::Ptr<cv::VideoCapture>())
                PROPERTY(cv::Ptr<cv::cudacodec::VideoReader>, d_cam, cv::Ptr<cv::cudacodec::VideoReader>())
                MO_SIGNAL(void, eos)
            MO_END;
            virtual bool LoadFile(const std::string& file_path);
            virtual bool d_LoadFile(const std::string& file_path);
            virtual bool h_LoadFile(const std::string& file_path);
            
            TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
            long long GetNumFrames();
        protected:
            virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
            virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
            TS<SyncedMemory>                        current_frame;
            bool got_frame = false;
        };
        class frame_grabber_camera: public frame_grabber_cv
        {
        public:
            frame_grabber_camera();
            MO_DERIVE(frame_grabber_camera, frame_grabber_cv)
                PROPERTY(long long, current_timestamp, 0)
            MO_END;

            static std::vector<std::string> ListLoadableDocuments();
            static int CanLoadDocument(const std::string& doc);
            virtual bool LoadFile(const std::string& file_path);
            virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
            rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
        protected:
        };
    }
}
