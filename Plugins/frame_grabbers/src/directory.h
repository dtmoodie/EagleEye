#pragma once

#include "EagleLib/IFrameGrabber.hpp"
#include "EagleLib/ICoordinateManager.h"

namespace EagleLib
{
    namespace Nodes
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
        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
        MO_BEGIN(frame_grabber_directory, IFrameGrabber)
            MO_SLOT(void, Restart);
        MO_END;

    private:
        cv::cuda::GpuMat                d_image;
        cv::Mat                         h_image;
        rcc::shared_ptr<ICoordinateManager>  coordinate_manager;
        std::string                     loaded_file;
        std::vector<std::string>        files_on_disk;
        int frame_index;
        boost::circular_buffer<std::tuple<std::string, TS<SyncedMemory>>> loaded_images;
        rcc::shared_ptr<IFrameGrabber> fg; // internal frame grabber used for loading the actual files
    };
    }
}