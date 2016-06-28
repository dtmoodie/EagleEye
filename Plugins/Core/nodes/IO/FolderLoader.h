#pragma once

#include <EagleLib/nodes/Node.h>
#include <EagleLib/utilities/CudaUtils.hpp>

namespace EagleLib
{
    namespace Nodes
    {
    
    class FolderLoader : public Node
    {
        void backgroundThread(boost::filesystem::path path);
        boost::thread thread;
        concurrent_notifier<cv::cuda::GpuMat> imageNotifier;
        void onDirectoryChange();
    public:
        FolderLoader();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    } 
}