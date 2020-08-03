#pragma once
#include "Aquila/utilities/cuda/CudaUtils.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class FFT : public Node
        {
          public:
            MO_DERIVE(FFT, Node)
                INPUT(SyncedImage, input)

                PARAM(bool, dft_rows, false)
                PARAM(bool, dft_scale, false)
                PARAM(bool, dft_inverse, false)
                PARAM(bool, dft_real_output, false)
                PARAM(bool, log_scale, true)
                PARAM(bool, use_optimized_size, false)

                OUTPUT(SyncedImage, magnitude)
                OUTPUT(SyncedImage, phase)
                OUTPUT(SyncedImage, coefficients)
            MO_END

          protected:
            virtual bool processImpl() override;
        };

        class FFTPreShiftImage : public Node
        {
            cv::cuda::GpuMat d_shiftMat;

          public:
            MO_DERIVE(FFTPreShiftImage, Node)
                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, output)
            MO_END

          protected:
            virtual bool processImpl() override;
        };

        class FFTPostShift : public Node
        {
            cv::cuda::GpuMat d_shiftMat;

          public:
            MO_DERIVE(FFTPostShift, Node)
                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, output)
            MO_END

          protected:
            virtual bool processImpl() override;
        };
    } // namespace nodes
} // namespace aq
