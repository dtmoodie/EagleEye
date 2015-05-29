#pragma once
#include "nodes/Node.h"
#include <CudaUtils.hpp>
#include <cudnn.h>
namespace EagleLib
{
    class ConvertToGrey: public Node
    {
    public:
        ConvertToGrey();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class ConvertToHSV: public Node
    {
    public:
        ConvertToHSV();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class ExtractChannels: public Node
    {
        int channelNum;
        ConstBuffer<std::vector<cv::cuda::GpuMat>> channelsBuffer;
        //std::vector<cv::cuda::GpuMat> channels;
    public:
        ExtractChannels();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class ConvertDataType: public Node
    {
    public:
        ConvertDataType();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class Merge: public Node
    {
        cv::cuda::GpuMat mergedChannels;
    public:
        Merge();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
}
