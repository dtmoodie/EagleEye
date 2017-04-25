#pragma once
#include "Aquila/Nodes/IFrameGrabber.hpp"

namespace aq
{
    namespace Nodes
    {
    /*class frame_grabber_directory: public IFrameGrabber
    {
    public:
        frame_grabber_directory();
        virtual bool LoadFile(const std::string& file_path);
        
        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
        
        MO_DERIVE(frame_grabber_directory, IFrameGrabber)
            MO_SLOT(void, Restart)
            STATUS(int, frame_index, 0)
            MO_SIGNAL(void, eos)
        MO_END;
        
        static int CanLoad(const std::string& doc);
    protected:
        bool ProcessImpl();
    private:
        cv::cuda::GpuMat                d_image;
        cv::Mat                         h_image;
        std::string                     loaded_file;
        std::vector<std::string>        files_on_disk;
        //int frame_index;
        boost::circular_buffer<std::tuple<std::string, TS<SyncedMemory>>> loaded_images;
        rcc::shared_ptr<IFrameGrabber> fg; // internal frame grabber used for loading the actual files
    };*/
    }
}
