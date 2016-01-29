#pragma once

#include "EagleLib/frame_grabber_base.h"
#include "EagleLib/ICoordinateManager.h"

namespace EagleLib
{
    class frame_grabber_image: public IFrameGrabber
    {
    public:

        class frame_grabber_image_info: public FrameGrabberInfo
        {
        public:
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
            virtual bool CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
        };
        virtual bool LoadFile(const std::string& file_path);
        virtual int GetFrameNumber();
        virtual int GetNumFrames();
        virtual std::string GetSourceFilename();

        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);

        virtual shared_ptr<ICoordinateManager> GetCoordinateManager();
    private:
        cv::cuda::GpuMat                d_image;
        cv::Mat                         h_image;
        shared_ptr<ICoordinateManager>  coordinate_manager;
        std::string                     loaded_file;
    };
}