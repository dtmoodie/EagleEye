#pragma once

#include "nodes/Node.h"
#include "CudaUtils.hpp"
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
    class Mat2Tensor: public Node
    {
        cv::cuda::GpuMat positionMat;
        BufferPool<cv::cuda::GpuMat, EventPolicy> bufferPool;
        BufferPool<cv::cuda::GpuMat, EventPolicy> convertedTypeBufferPool;
    public:
        Mat2Tensor();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
        virtual void Init(bool firstInit);
    };
    class ConcatTensor: public Node
    {
        BufferPool<cv::cuda::GpuMat, EventPolicy> d_buffer;
    public:
        ConcatTensor();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
        virtual void Init(bool firstInit);
    };
}
