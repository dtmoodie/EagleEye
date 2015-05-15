#pragma once

#include "nodes/Node.h"

namespace EagleLib
{
    class GetOutputImage: public Node
    {
    public:
        GetOutputImage();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void Init(bool firstInit);
    };
    class ExportInputImage: public Node
    {
    public:
        ExportInputImage();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void Init(bool firstInit);
    };

    class ImageInfo: public Node
    {
    public:
        ImageInfo();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
        virtual void Init(bool firstInit);
    };
}
