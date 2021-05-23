#pragma once
#include "Aquila/nodes/Node.hpp"
#include "Aquila/detail/export.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include <Aquila/utilities/cuda/CudaUtils.hpp>
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"

#define FLANN_USE_CUDA
#include "flann/flann.hpp"

RUNTIME_COMPILER_LINKLIBRARY("cudart_static.lib")
RUNTIME_COMPILER_LINKLIBRARY("cublas.lib")
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("flann_cpp_sd.lib")
RUNTIME_COMPILER_LINKLIBRARY("flann_cuda_sd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("flann_cpp_s.lib")
RUNTIME_COMPILER_LINKLIBRARY("flann_cuda_s.lib")
#endif

namespace aq
{
    namespace nodes
    {
        class ForegroundEstimate : public Node
        {
        public:
            MO_DERIVE(ForegroundEstimate, Node)
                INPUT(SyncedMemory, input_point_cloud, nullptr)
                PARAM(float, radius, 5.0)
                PARAM(float, epsilon, 1.0)
                PARAM(int, checks, -1)
                PARAM(bool, build_model, false)
                OUTPUT(SyncedMemory, background_model, SyncedMemory())
                APPEND_FLAGS(background_model, mo::Unstamped_e)
                OUTPUT(SyncedMemory, index, SyncedMemory())
                OUTPUT(SyncedMemory, distance, SyncedMemory())
                OUTPUT(SyncedMemory, point_mask, SyncedMemory())
                OUTPUT(SyncedMemory, foreground, SyncedMemory())
            MO_END;
        protected:
            bool processImpl();
            void BuildModel(cv::cuda::GpuMat& tensor, cv::cuda::Stream& stream);
            std::shared_ptr<flann::GpuIndex<flann::L2<float>>> nnIndex;
        };
    }
}
