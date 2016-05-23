#pragma once

#include "EagleLib/nodes/Node.h"
#include <EagleLib/Defs.hpp>
#include <EagleLib/utilities/CudaUtils.hpp>
#include <EagleLib/Project_defs.hpp>
#include "RuntimeLinkLibrary.h"


#define FLANN_USE_CUDA
#include "flann/flann.hpp"
SETUP_PROJECT_DEF;

RUNTIME_COMPILER_LINKLIBRARY("cudart_static.lib")
RUNTIME_COMPILER_LINKLIBRARY("cublas.lib")
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("flann_cpp_sd.lib")
RUNTIME_COMPILER_LINKLIBRARY("flann_cuda_sd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("flann_cpp_s.lib")
RUNTIME_COMPILER_LINKLIBRARY("flann_cuda_s.lib")
#endif

namespace EagleLib
{
    namespace Nodes
    {
    
	class ForegroundEstimate : public Node
	{
		bool _build_model;
		void BuildModel(cv::cuda::GpuMat& tensor, cv::cuda::Stream& stream);
		std::shared_ptr<flann::GpuIndex<flann::L2<float>>> nnIndex;
	public:
		ForegroundEstimate();
		virtual void NodeInit(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
    }
}
