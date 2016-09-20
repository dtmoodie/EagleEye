#pragma once
#include "EagleLib/Nodes/IFrameGrabber.hpp"
#include "EagleLib/ICoordinateManager.h"

namespace EagleLib
{
    namespace Nodes
    {
    class PLUGIN_EXPORTS frame_grabber_directory: public IFrameGrabber
    {
    public:
        frame_grabber_directory();
        virtual bool LoadFile(const std::string& file_path);
        virtual long long GetFrameNumber();
        virtual long long GetNumFrames();
        virtual std::string GetSourceFilename();

        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream);

        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
        
        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
        
        MO_DERIVE(frame_grabber_directory, IFrameGrabber)
            MO_SLOT(void, Restart);
        MO_END;
        
        static int CanLoadDocument(const std::string& doc);
        
    protected:
        bool ProcessImpl();


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