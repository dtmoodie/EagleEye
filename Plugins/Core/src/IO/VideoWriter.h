#pragma once
#include <src/precompiled.hpp>
#include <Aquila/rcc/external_includes/cv_videoio.hpp>
#include <Aquila/rcc/external_includes/cv_cudacodec.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include "MetaObject/Thread/ThreadHandle.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace Nodes
    {
    
    class VideoWriter : public Node
    {
    public:
        MO_DERIVE(VideoWriter, Node)
            INPUT(SyncedMemory, image, nullptr)
            PROPERTY(cv::Ptr<cv::cudacodec::VideoWriter>, d_writer, cv::Ptr<cv::cudacodec::VideoWriter>())
            PROPERTY(cv::Ptr<cv::VideoWriter>, h_writer, cv::Ptr<cv::VideoWriter>())
            PARAM(mo::EnumParameter, codec, mo::EnumParameter())
            PARAM(mo::WriteFile, filename, mo::WriteFile("video.avi"))
            PARAM(bool, using_gpu_writer, true)
            MO_SLOT(void, write_out)
            PROPERTY(mo::ThreadHandle, _write_thread, mo::ThreadPool::Instance()->RequestThread())
        MO_END;
        void NodeInit(bool firstInit);
    protected:
        bool ProcessImpl();
        //mo::ThreadHandle _write_thread;
        
    };
    }
}
