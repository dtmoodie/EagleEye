#pragma once
#include "nodes/Node.h"

namespace EagleLib
{
    class ConvertToGrey: public Node
    {
    public:
        ConvertToGrey();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };

    class ConvertToHSV: public Node
    {
    public:
        ConvertToHSV();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };

    class ExtractChannels: public Node
    {
        int channelNum;
    public:
        ExtractChannels();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };
}
