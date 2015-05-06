#pragma once

#include <nodes/Node.h>


namespace EagleLib
{

    class SetDevice: public Node
    {
        bool firstRun;
    public:
        SetDevice();
        virtual bool SkipEmpty() const;
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };


}
