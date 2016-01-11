#pragma once
#include "Defs.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>

namespace EagleLib
{
	class EAGLE_EXPORTS SyncedMemory
	{
		enum
		{
			SYNCED = 0,
			HOST_UPDATED,
			DEVICE_UPDATED
		};
		std::vector<cv::Mat> h_data;
		std::vector<cv::cuda::GpuMat> d_data;
		std::vector<int> sync_flags;
	public:
		SyncedMemory();
		SyncedMemory(cv::MatAllocator* cpu_allocator, cv::cuda::GpuMat::Allocator* gpu_allocator);

		const cv::Mat&				GetMat(cv::cuda::Stream& stream, int = 0);
		cv::Mat&					GetMatMutable(cv::cuda::Stream& stream, int = 0);

		const cv::cuda::GpuMat&		GetGpuMat(cv::cuda::Stream& stream, int = 0);
		cv::cuda::GpuMat&			GetGpuMatMutable(cv::cuda::Stream& stream, int = 0);

		const std::vector<cv::Mat>&				GetMatVec(cv::cuda::Stream& stream);
		std::vector<cv::Mat>&		GetMatVecMutable(cv::cuda::Stream& stream);

		const std::vector<cv::cuda::GpuMat>&		GetGpuMatVec(cv::cuda::Stream& stream);
		std::vector<cv::cuda::GpuMat>&			GetGpuMatVecMutable(cv::cuda::Stream& stream);
		int GetNumMats() const;
		void ResizeNumMats();
	};
}