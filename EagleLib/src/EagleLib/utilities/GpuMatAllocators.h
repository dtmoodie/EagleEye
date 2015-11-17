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
	class EAGLE_EXPORTS PitchedAllocator : public cv::cuda::GpuMat::Allocator
	{
	public:
		PitchedAllocator();
		void SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride);
		void Increment(unsigned char* ptr, size_t size);
		void Decrement(unsigned char* ptr, size_t size);
		void SetScope(const std::string& name);
	protected:
		size_t textureAlignment;
		size_t memoryUsage;
		std::recursive_mutex mtx;
		std::map<std::string, size_t> scopedAllocationSize;
		std::string currentScopeName;
		std::map<unsigned char*, std::string> scopeOwnership;
	};

	class MemoryBlock;

	class EAGLE_EXPORTS BlockMemoryAllocator: public PitchedAllocator
	{
		std::list<std::shared_ptr<MemoryBlock>> blocks;
	public:
		BlockMemoryAllocator(size_t initialBlockSize);
		virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
		virtual void free(cv::cuda::GpuMat* mat);
		size_t initialBlockSize_;
	};

	class EAGLE_EXPORTS DelayedDeallocator : public PitchedAllocator
	{
	public:
		DelayedDeallocator();
		virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
		virtual void free(cv::cuda::GpuMat* mat);
		size_t deallocateDelay; // ms
		
	protected:
		virtual void clear();
		std::map<unsigned char*, std::pair<clock_t, size_t>> deallocateList; // list of all the different memory blocks to be deallocated
	};	
}