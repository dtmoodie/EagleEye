#pragma once

#include <nodes/Node.h>
#include "CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    class SetDevice: public Node
    {
        int currentDevice;
        unsigned int maxDevice;
        bool firstRun;
    public:
        SetDevice();
        virtual bool SkipEmpty() const;
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class StreamDispatcher: public Node
    {
        ConstBuffer<cv::cuda::Stream> streams;
    public:
        StreamDispatcher();
        virtual bool SkipEmpty() const;
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };


}
