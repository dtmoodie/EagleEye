#pragma once
#include <src/precompiled.hpp>
#include <EagleLib/rcc/external_includes/cv_videoio.hpp>
#include <EagleLib/rcc/external_includes/cv_cudacodec.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class VideoWriter : public Node
    {
    public:
        MO_DERIVE(VideoWriter, Node)
            INPUT(SyncedMemory, image, nullptr);
            PROPERTY(cv::Ptr<cv::cudacodec::VideoWriter>, d_writer, cv::Ptr<cv::cudacodec::VideoWriter>())
            PROPERTY(cv::Ptr<cv::VideoWriter>, h_writer, cv::Ptr<cv::VideoWriter>())
            PARAM(mo::EnumParameter, codec, mo::EnumParameter());
            PARAM(mo::WriteFile, filename, mo::WriteFile("video.mp4"));
            STATUS(bool, using_gpu_writer, true);
            MO_SLOT(void, write_out);
        MO_END;

    protected:
        bool ProcessImpl();
        
    };
    }
}
