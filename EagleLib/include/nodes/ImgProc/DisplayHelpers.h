#pragma once

#include "nodes/Node.h"


namespace EagleLib
{
    class AutoScale: public Node
    {
    public:
        AutoScale();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };

    class Colormap: public Node
    {
    public:
        Colormap();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };


}
