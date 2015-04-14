#pragma once

#include "nodes/Node.h"

namespace EagleLib
{
    class GetOutputImage: public Node
    {
    public:
        GetOutputImage();
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
        void Init(bool firstInit);
    };
}
