#pragma once
#include "Node.h"
namespace EagleLib
{
	class EAGLE_EXPORTS CpuExtraction : public Node
	{
	public:
		virtual void process(SyncedMemory& input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual void processImpl(const cv::Mat& mat, cv::cuda::Stream& stream) = 0;
	};

	class EAGLE_EXPORTS GpuExtraction : public Node
	{
	public:
		virtual void process(SyncedMemory& input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual void processImpl(const cv::cuda::GpuMat& mat, cv::cuda::Stream& stream) = 0;
	};
}