#pragma once
#include "Node.h"
namespace EagleLib
{
	class EAGLE_EXPORTS CpuProcessing: public Node
	{
	public:
		virtual void process(SyncedMemory& input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual void processImpl(cv::Mat& mat, cv::cuda::Stream& stream) = 0;
	};
	class EAGLE_EXPORTS GpuProcessing: public Node
	{
		virtual void process(SyncedMemory& input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual void processImpl(cv::cuda::GpuMat& mat, cv::cuda::Stream& stream) = 0;
	};
}