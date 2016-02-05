#pragma once
#include "Defs.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>

namespace EagleLib
{
    template<typename T> class TS: public T
    {
    public:
        template<class...U> TS(U...args):T(args...)
        {
            timestamp = 0.0;
            frame_number = 0.0;
        }
        template<class...U> TS(double ts, int frame_number, U...args) : T(args...)
        {
            timestamp = ts;
            this->frame_number = frame_number;
        }
        double timestamp;
        int frame_number;
    };


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
        SyncedMemory(const cv::Mat& h_mat, const cv::cuda::GpuMat& d_mat);
        SyncedMemory(const cv::Mat& h_mat);
        SyncedMemory(const cv::cuda::GpuMat& d_mat);
        SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat);
		SyncedMemory(cv::MatAllocator* cpu_allocator, cv::cuda::GpuMat::Allocator* gpu_allocator);
        SyncedMemory clone(cv::cuda::Stream& stream);

		const cv::Mat&				GetMat(cv::cuda::Stream& stream, int = 0);
		cv::Mat&					GetMatMutable(cv::cuda::Stream& stream, int = 0);

		const cv::cuda::GpuMat&		GetGpuMat(cv::cuda::Stream& stream, int = 0);
		cv::cuda::GpuMat&			GetGpuMatMutable(cv::cuda::Stream& stream, int = 0);

		const std::vector<cv::Mat>&				GetMatVec(cv::cuda::Stream& stream);
		std::vector<cv::Mat>&		GetMatVecMutable(cv::cuda::Stream& stream);

		const std::vector<cv::cuda::GpuMat>&		GetGpuMatVec(cv::cuda::Stream& stream);
		std::vector<cv::cuda::GpuMat>&			GetGpuMatVecMutable(cv::cuda::Stream& stream);
		int GetNumMats() const;
        bool empty() const;
		void ResizeNumMats(int new_size = 1);
	};
}