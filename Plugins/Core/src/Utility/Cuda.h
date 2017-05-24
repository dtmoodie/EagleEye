#pragma once

#include <Aquila/nodes/Node.hpp>
#include "Aquila/utilities/cuda/CudaUtils.hpp"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
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
        virtual void nodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    }
}
