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
        static void applyLUT(thrust::device_vector<cv::Vec3b> d_LUT, cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        ColorScale red, green, blue;
        //thrust::device_vector<cv::Vec3b> d_LUT;
        std::vector<cv::Vec3b> LUT;
        size_t resolution;
        double scale, shift;
        void buildLUT();
    public:
        Colormap();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class QtColormapDisplay: public Colormap
    {

        BufferPool<cv::cuda::HostMem, EventPolicy> h_bufferPool;
        BufferPool<cv::cuda::GpuMat, EventPolicy> d_scaledBufferPool;
    public:
        void display();
        QtColormapDisplay();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

//    class Colormap: public Node
//    {
//        static void applyLUT(thrust::device_vector<cv::Vec3b> d_LUT, cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
//        ColorScale red, green, blue;
//        //thrust::device_vector<cv::Vec3b> d_LUT;
//        std::vector<cv::Vec3b> LUT;
//        size_t resolution;
//        double scale, shift;
//        void buildLUT();
//        Buffer<cv::cuda::HostMem, EventPolicy>* h_buffer;
//        Buffer<cv::cuda::GpuMat, EventPolicy>* d_buffer;
//        BufferPool<cv::cuda::HostMem, EventPolicy> h_bufferPool;
//        BufferPool<cv::cuda::GpuMat, EventPolicy> d_bufferPool;
//        BufferPool<cv::cuda::GpuMat> d_scaledBufferPool;
//    public:
//        Colormap();
//        virtual void applyColormap();
//        virtual void Init(bool firstInit);
//        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
//    };
    class Normalize: public Node
    {
        ConstBuffer<cv::cuda::GpuMat> normalizedBuf;
    public:
        Normalize();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
