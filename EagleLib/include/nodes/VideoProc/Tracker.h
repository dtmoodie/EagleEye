#pragma once

#include <nodes/Node.h>

namespace EagleLib
{
    class KeyFrameTracker: public Node
    {

    public:
        KeyFrameTracker();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());


    };

    class CMTTracker: public Node
    {

    public:
        CMTTracker();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    };

    class TLDTracker:public Node
    {
    public:
        TLDTracker();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
