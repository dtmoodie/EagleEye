#pragma once

#include <nodes/Node.h>

namespace EagleLib
{
    class FFT: public Node
    {
    public:
        FFT();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };
}
