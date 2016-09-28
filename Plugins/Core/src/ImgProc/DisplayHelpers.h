#pragma once

#include "EagleLib/nodes/Node.h"
#include <EagleLib/utilities/CudaUtils.hpp>
#include <EagleLib/utilities/ColorMapping.hpp>
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    class AutoScale: public Node
    {
    public:
        AutoScale();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    /*class Colormap: public Node
    {
    protected:
        cv::cuda::GpuMat color_mapped_image;
        color_mapper mapper;
    public:
        void Rescale();
        bool rescale;
        Colormap();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };*/
    /*class QtColormapDisplay: public Colormap
    {
    public:
        void display();
        QtColormapDisplay();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };*/
    class Normalize: public Node
    {
    public:
        Normalize();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    }
}
