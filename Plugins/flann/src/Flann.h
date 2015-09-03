#pragma once

#include "nodes/Node.h"
#include <CudaUtils.hpp>


#define FLANN_USE_CUDA
#include "flann/flann.hpp"
SETUP_PROJECT_DEF
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
	class PtCloud_backgroundSubtract_flann : public Node
	{
		//flann::Matrix<float> input;
		cv::cuda::GpuMat input;
		cv::cuda::GpuMat result;
		BufferPool<cv::cuda::GpuMat> inputBuffer;
		BufferPool<cv::cuda::GpuMat> idxBuffer;
		BufferPool<cv::cuda::GpuMat> distBuffer;
		BufferPool<GpuResized<cv::cuda::GpuMat>> outputBuffer;
		cv::cuda::GpuMat count;
		void BuildModel();
		bool MapInput(cv::cuda::GpuMat& img = cv::cuda::GpuMat());

		std::shared_ptr<flann::GpuIndex<flann::L2<float>>> nnIndex;
	public:
		PtCloud_backgroundSubtract_flann();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
}