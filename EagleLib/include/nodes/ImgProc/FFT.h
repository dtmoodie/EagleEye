#pragma once

#include <nodes/Node.h>

namespace EagleLib
{
    class FFT: public Node
    {
    public:
        FFT();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };

    class FFTPreShiftImage: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        FFTPreShiftImage();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };

    class FFTPostShift: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        FFTPostShift();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };
}
