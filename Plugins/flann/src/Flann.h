#pragma once

#include "nodes/Node.h"
#include <CudaUtils.hpp>

#define FLANN_USE_CUDA
#include "flann/flann.hpp"

namespace EagleLib
{
	class PtCloud_backgroundSubtract_flann : public Node
	{
		BufferPool<cv::cuda::GpuMat> inputBuffer;
		BufferPool<cv::cuda::GpuMat> idxBuffer;
		BufferPool<cv::cuda::GpuMat> distBuffer;

		std::shared_ptr<flann::GpuIndex<flann::L2<float>>> nnIndex;
	public:
		PtCloud_backgroundSubtract_flann();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
}