#pragma once
#include <EagleLib/Defs.hpp>
#include <opencv2/core/cuda.hpp>
#include <mutex>
#include <map>
#include <list>
#include <memory>

namespace EagleLib
{
	cv::cuda::GpuMat::Allocator* GetDefaultBlockMemoryAllocator();
	cv::cuda::GpuMat::Allocator* GetDefaultDelayedDeallocator();
	cv::cuda::GpuMat::Allocator* CreateBlockMemoryAllocator();
	cv::cuda::GpuMat::Allocator* CreateBlockMemoryAllocator();
	class PitchedAllocator : public cv::cuda::GpuMat::Allocator
	{
	public:
		PitchedAllocator();

	protected:
		size_t textureAlignment;
		size_t memoryUsage;
		std::recursive_mutex mtx;
	};

	class MemoryBlock;

	class BlockMemoryAllocator: public PitchedAllocator
	{
		std::list<std::shared_ptr<MemoryBlock>> blocks;
	public:
		BlockMemoryAllocator(size_t initialBlockSize);
		virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
		virtual void free(cv::cuda::GpuMat* mat);
		size_t initialBlockSize_;
	};

	class DelayedDeallocator : public PitchedAllocator
	{
	public:
		virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
		virtual void free(cv::cuda::GpuMat* mat);
		size_t deallocateDelay; // ms
	private:
		void clear();
		std::map<unsigned char*, std::pair<clock_t, size_t>> deallocateList; // list of all the different memory blocks to be deallocated
	};
}