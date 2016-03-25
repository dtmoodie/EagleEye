#pragma once

#include "EagleLib/frame_grabber_base.h"
#include "EagleLib/ICoordinateManager.h"

namespace EagleLib
{
    class PLUGIN_EXPORTS frame_grabber_directory: public IFrameGrabber
    {
    public:

        class PLUGIN_EXPORTS frame_grabber_directory_info: public FrameGrabberInfo
        {
        public:
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
            virtual int CanLoadDocument(const std::string& document) const;
            virtual int Priority() const;
        };
        frame_grabber_directory();
        virtual bool LoadFile(const std::string& file_path);
        virtual int GetFrameNumber();
        virtual int GetNumFrames();
        virtual std::string GetSourceFilename();

        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);
		virtual TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream);

        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
    private:
        cv::cuda::GpuMat                d_image;
        cv::Mat                         h_image;
        rcc::shared_ptr<ICoordinateManager>  coordinate_manager;
        std::string                     loaded_file;
        std::vector<std::string>        files_on_disk;
        int frame_index;
        boost::circular_buffer<std::tuple<std::string, cv::Mat, cv::cuda::GpuMat>> loaded_images;
        
    };
}