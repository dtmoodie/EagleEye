#pragma once
#include <src/precompiled.hpp>
#include <EagleLib/nodes/Node.h>
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class FFT: public Node
    {
    public:
        MO_DERIVE(FFT, Node)
            INPUT(SyncedMemory, input, nullptr);
            PARAM(bool, dft_rows, false);
            PARAM(bool, dft_scale, false);
            PARAM(bool, dft_inverse, false);
            PARAM(bool, dft_real_output, false);
            PARAM(bool, log_scale, true);
            PARAM(bool, use_optimized_size, false);
            OUTPUT(SyncedMemory, magnitude, SyncedMemory());
            OUTPUT(SyncedMemory, phase, SyncedMemory());
            OUTPUT(SyncedMemory, coefficients, SyncedMemory());
    protected:
        bool ProcessImpl();
    };

    class FFTPreShiftImage: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        FFTPreShiftImage();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class FFTPostShift: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        FFTPostShift();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    }
}
