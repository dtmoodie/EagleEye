#include "GpuMatAllocators.h"
#include <cuda_runtime.h>
#include <opencv2/cudev/common.hpp>
#include <boost/log/trivial.hpp>
#include <logger.hpp>

namespace EagleLib
{
	unsigned char* alignMemory(unsigned char* ptr, int elemSize)
	{
		int i;
		for (i = 0; i < elemSize; ++i)
		{
			if (reinterpret_cast<size_t>(ptr + i) % elemSize == 0)
			{
				break;
			}
		}
		return ptr + i;  // Forces memory to be aligned to an element's byte boundary
	}
	int alignmentOffset(unsigned char* ptr, int elemSize)
	{
		int i;
		for (i = 0; i < elemSize; ++i)
		{
			if (reinterpret_cast<size_t>(ptr + i) % elemSize == 0)
			{
				break;
			}
		}
		return i;
	}
	void PitchedAllocator::SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride)
	{
		if (rows == 1 || cols == 1)
		{
			stride = cols*elemSize;
		}
		else
		{
			stride = cols*elemSize + textureAlignment - (cols*elemSize % textureAlignment);
		}
		sizeNeeded = stride*rows;
	}
	void PitchedAllocator::SetScope(const std::string& name)
	{
		auto itr = scopedAllocationSize.find(name);
		if (itr == scopedAllocationSize.end())
		{
			scopedAllocationSize[name] = 0;
			currentScopeName = name; 
		}
		currentScopeName = name;
	}
	PitchedAllocator::PitchedAllocator()
	{
		textureAlignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
		memoryUsage = 0;
		SetScope("Default");
	}
	void PitchedAllocator::Increment(unsigned char* ptr, size_t size)
	{
		scopeOwnership[ptr] = currentScopeName;
		scopedAllocationSize[currentScopeName] += size;
	}
	void PitchedAllocator::Decrement(unsigned char* ptr, size_t size)
	{
		auto itr = scopeOwnership.find(ptr);
		if (itr != scopeOwnership.end())
		{
			scopedAllocationSize[itr->second] -= size;
		}		
	}
	struct MemoryBlock
	{
		MemoryBlock(size_t size_):size(size_)
		{
			CV_CUDEV_SAFE_CALL(cudaMalloc(&begin, size)); end = begin + size;		
		}
		unsigned char* allocate(size_t size_, size_t elemSize_)
		{
			if (size_ > size)
				return nullptr;
			std::vector<std::pair<size_t, unsigned char*>> candidates;
			unsigned char* prevEnd = begin;
			if (allocatedBlocks.size())
			{
				for (auto itr : allocatedBlocks)
				{
					if (static_cast<size_t>(itr.first - prevEnd) > size_)
					{
						auto alignment = alignmentOffset(prevEnd, elemSize_);
						if (static_cast<size_t>(itr.first - prevEnd + alignment) > size_)
						{
							candidates.push_back(std::make_pair(size_t(itr.first - prevEnd + alignment), prevEnd + alignment));
						}
					}
					prevEnd = itr.second;
				}
			}
			if (static_cast<size_t>(end - prevEnd) > size_)
			{
				auto alignment = alignmentOffset(prevEnd, elemSize_);
				if (static_cast<size_t>(end - prevEnd + alignment) > size_)
				{
					candidates.push_back(std::make_pair(size_t(end - prevEnd + alignment), prevEnd + alignment));
				}
			}
			// Find the smallest chunk of memory that fits our requirement, helps reduce fragmentation.
			auto min = std::min_element(candidates.begin(), candidates.end(), [](const std::pair<size_t, unsigned char*>& first, const std::pair<size_t, unsigned char*>& second) {return first.first < second.first; });
			if (min != candidates.end() && min->first > size_)
			{
				allocatedBlocks[min->second] = (unsigned char*)(min->second + size_);
				return min->second;
			}
			return nullptr;

		}
		bool deAllocate(unsigned char* ptr)
		{
			if(ptr < begin || ptr > end)
				return false;
			auto itr = allocatedBlocks.find(ptr);
			if (itr != allocatedBlocks.end())
			{
				allocatedBlocks.erase(itr);
				return true;
			}
			return true;
		}
		unsigned char* begin;
		unsigned char* end;
		size_t size;
		std::map<unsigned char*, unsigned char*> allocatedBlocks;
	};

	BlockMemoryAllocator::BlockMemoryAllocator(size_t initialSize)
	{
		blocks.push_back(std::shared_ptr<MemoryBlock>(new MemoryBlock(initialSize)));
		initialBlockSize_ = initialSize;
	}

	bool BlockMemoryAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
	{
		std::lock_guard<std::recursive_mutex> lock(mtx);
		size_t sizeNeeded, stride;
		SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
		unsigned char* ptr;
		for (auto itr : blocks)
		{
			ptr = itr->allocate(sizeNeeded, elemSize);
			if (ptr)
			{
				mat->data = ptr;
				mat->step = stride;
				mat->refcount = mat->refcount = (int*)cv::fastMalloc(sizeof(int));
				memoryUsage += mat->step*rows;
				BOOST_LOG_TRIVIAL(debug) << "Allocating block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. total usage: " << memoryUsage / (1024 * 1024) << " MB";
				Increment(ptr, mat->step*rows);
				return true;
			}
		}
		// If we get to this point, then no memory was found, need to allocate new memory
		blocks.push_back(std::shared_ptr<MemoryBlock>(new MemoryBlock(std::max(initialBlockSize_ / 2, sizeNeeded))));
		BOOST_LOG_TRIVIAL(warning) << "Expanding memory pool by " <<  std::max(initialBlockSize_ / 2, sizeNeeded) / (1024 * 1024) << " MB";
		if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, elemSize))
		{
			mat->data = ptr;
			mat->step = stride;
			mat->refcount = mat->refcount = (int*)cv::fastMalloc(sizeof(int));
			memoryUsage += mat->step*rows;
			Increment(ptr, mat->step*rows);
			BOOST_LOG_TRIVIAL(debug) << "Allocating block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. total usage: " << memoryUsage / (1024 * 1024) << " MB";
			return true;
		}
		return false;
	}
	void BlockMemoryAllocator::free(cv::cuda::GpuMat* mat)
	{
		std::lock_guard<std::recursive_mutex> lock(mtx);
		for (auto itr : blocks)
		{
			if (itr->deAllocate(mat->data))
			{
				cv::fastFree(mat->refcount);
				return;
			}
		}
		throw cv::Exception(0, "Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
	}
	DelayedDeallocator::DelayedDeallocator() :PitchedAllocator()
	{

	}
	
	bool DelayedDeallocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
	{
		// First check for anything of the correct size
		std::lock_guard<std::recursive_mutex> lock(mtx);
		size_t sizeNeeded, stride;
		SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
		for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
		{
			if (itr->second.second == sizeNeeded)
			{
				mat->data = itr->first;
				mat->step = stride;
				mat->refcount = mat->refcount = (int*)cv::fastMalloc(sizeof(int));
				deallocateList.erase(itr);
				memoryUsage += mat->step*rows;
				Increment(mat->data, mat->step*rows);
				BOOST_LOG_TRIVIAL(debug) << "Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. total usage: " << memoryUsage / (1024 * 1024) << " MB";
				return true;
			}
		}
		if (rows > 1 && cols > 1)
		{
			CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
			memoryUsage += mat->step*rows;
			Increment(mat->data, mat->step*rows);
			BOOST_LOG_TRIVIAL(info) << "Allocating block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
		}
		else
		{
			CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
			memoryUsage += elemSize*cols*rows;
			Increment(mat->data, elemSize*cols*rows);
			BOOST_LOG_TRIVIAL(info) << "Allocating block of size (" << rows << "," << cols << ") " << cols * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
			mat->step = elemSize * cols;
		}
		mat->refcount = (int*)cv::fastMalloc(sizeof(int));
		return true;
	}
	void DelayedDeallocator::free(cv::cuda::GpuMat* mat)
	{
		std::lock_guard<std::recursive_mutex> lock(mtx);
		Decrement(mat->data, mat->step*mat->rows);
		scopedAllocationSize[scopeOwnership[mat->data]] -= mat->rows*mat->step;
		BOOST_LOG_TRIVIAL(debug) << "Releasing mat of size (" << mat->rows << "," << mat->cols << ") " << (mat->dataend - mat->datastart)/(1024*1024) << " MB to the memory pool";
		deallocateList[mat->data] = std::pair<clock_t, size_t>(clock(), mat->rows*mat->step);
		cv::fastFree(mat->refcount);
		clear();
	}
	void DelayedDeallocator::clear()
	{
		auto time = clock();
		for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
		{
			if ((time - itr->second.first) > deallocateDelay)
			{
				memoryUsage -= itr->second.second;
				BOOST_LOG_TRIVIAL(info) << "Deallocating block of size " << itr->second.second/(1024*1024) << "MB. Which was stale for " << time - itr->second.first << " ms";
				CV_CUDEV_SAFE_CALL(cudaFree(itr->first));
				itr = deallocateList.erase(itr);
			}
		}
	}
}
