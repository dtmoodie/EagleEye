#pragma once

#include <nodes/Node.h>
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    class FFT: public Node
    {
        enum output
        {
            Coefficients = -1,
            Magnitude = 0,
            Phase = 1
        };
        ConstBuffer<cv::cuda::GpuMat> destBuf;
        ConstBuffer<cv::cuda::GpuMat> floatBuf;
    public:
        FFT();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class FFTPreShiftImage: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        FFTPreShiftImage();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class FFTPostShift: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        FFTPostShift();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
