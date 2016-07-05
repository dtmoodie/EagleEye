#pragma once

#include "EagleLib/nodes/Node.h"
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class GetOutputImage: public Node
    {
    public:
        GetOutputImage();
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void NodeInit(bool firstInit);
    };
    class ExportInputImage: public Node
    {
    public:
        ExportInputImage();
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void NodeInit(bool firstInit);
    };

    class ImageInfo: public Node
    {
    public:
        ImageInfo();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
        virtual void NodeInit(bool firstInit);
    };
    class Mat2Tensor: public Node
    {
        cv::cuda::GpuMat positionMat;
    public:
        Mat2Tensor();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
        virtual void NodeInit(bool firstInit);
    };
    class ConcatTensor: public Node
    {
        BufferPool<cv::cuda::GpuMat, EventPolicy> d_buffer;
    public:
        ConcatTensor();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
        virtual void NodeInit(bool firstInit);
    };
    class LagBuffer : public Node
    {
        std::vector<cv::cuda::GpuMat> imageBuffer;
        unsigned int putItr;
        unsigned int getItr;
        unsigned int lagFrames;
    public:
        LagBuffer();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
        virtual void NodeInit(bool firstInit);
    };

    class CameraSync : public Node
    {
    public:
        CameraSync();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
        virtual void NodeInit(bool firstInit);
        bool SkipEmpty() const;
    };
    } // namespace nodes
}
