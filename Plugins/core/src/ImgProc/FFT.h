#pragma once
#include <src/precompiled.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include "Aquila/utilities/cuda/CudaUtils.hpp"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace nodes
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
        MO_END;
    protected:
        bool processImpl();
    };

    class FFTPreShiftImage: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        MO_DERIVE(FFTPreShiftImage, Node);
            INPUT(SyncedMemory, input, nullptr);
            OUTPUT(SyncedMemory, output, SyncedMemory());
        MO_END;
    protected:
        bool processImpl();
    };

    class FFTPostShift: public Node
    {
        cv::cuda::GpuMat d_shiftMat;
    public:
        MO_DERIVE(FFTPostShift, Node);
            INPUT(SyncedMemory, input, nullptr);
            OUTPUT(SyncedMemory, output, SyncedMemory());
        MO_END;
    protected:
        bool processImpl();
        //FFTPostShift();
        //virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    }
}
