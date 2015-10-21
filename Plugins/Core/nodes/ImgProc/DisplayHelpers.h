#pragma once

#include "nodes/Node.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <CudaUtils.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#include "DisplayHelpers.cuh"

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
