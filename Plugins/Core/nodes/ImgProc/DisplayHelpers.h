#pragma once

#include "nodes/Node.h"
#include <CudaUtils.hpp>
#include "DisplayHelpers.cuh"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    class AutoScale: public Node
    {
    public:
        AutoScale();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class Colormap: public Node
    {
    protected:
		cv::cuda::GpuMat color_mapped_image;
		color_mapper mapper;
    public:
		void Rescale();
		bool rescale;
        Colormap();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class QtColormapDisplay: public Colormap
    {
    public:
        void display();
        QtColormapDisplay();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class Normalize: public Node
    {
        ConstBuffer<cv::cuda::GpuMat> normalizedBuf;
    public:
        Normalize();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
