#pragma once

#include <EagleLib/nodes/Node.h>
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    class SetDevice: public Node
    {
        int currentDevice;
        unsigned int maxDevice;
        bool firstRun;
    public:
        SetDevice();
        virtual bool SkipEmpty() const;
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class StreamDispatcher: public Node
    {
        ConstBuffer<cv::cuda::Stream> streams;
    public:
        StreamDispatcher();
        virtual bool SkipEmpty() const;
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    }
}
