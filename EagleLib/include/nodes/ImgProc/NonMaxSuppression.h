#pragma once

#include "nodes/Node.h"


namespace EagleLib
{
    class NonMaxSuppression: public Node
    {
    public:
        NonMaxSuppression();
        void Init(bool firstInit);
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };
}
