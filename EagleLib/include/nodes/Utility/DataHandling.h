#pragma once

#include "nodes/Node.h"

namespace EagleLib
{
    class GetOutputImage: public Node
    {
    public:
        GetOutputImage();
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
        void Init(bool firstInit);
    };
}
