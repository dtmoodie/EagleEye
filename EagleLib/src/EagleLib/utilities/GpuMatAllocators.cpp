#include "GpuMatAllocators.h"
#include <boost/log/trivial.hpp>
#include <logger.hpp>
#include "MemoryBlock.h"
#include <opencv2/cudev/common.hpp>
#include <cuda_runtime.h>
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
	

	BlockMemoryAllocator::BlockMemoryAllocator(size_t initialSize)
	{
		blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(initialSize)));
		initialBlockSize_ = initialSize;
	}
	BlockMemoryAllocator* BlockMemoryAllocator::Instance(size_t initial_size)
	{
		static BlockMemoryAllocator* inst = nullptr;
		if (inst == nullptr)
		{
			inst = new BlockMemoryAllocator(initial_size);
		}
		return inst;
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
                mat->refcount = (int*)cv::fastMalloc(sizeof(int));
				memoryUsage += mat->step*rows;
				BOOST_LOG_TRIVIAL(debug) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
				Increment(ptr, mat->step*rows);
				return true;
			}
		}
		// If we get to this point, then no memory was found, need to allocate new memory
		blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(initialBlockSize_ / 2, sizeNeeded))));
		BOOST_LOG_TRIVIAL(warning) << "[GPU] Expanding memory pool by " <<  std::max(initialBlockSize_ / 2, sizeNeeded) / (1024 * 1024) << " MB";
		if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, elemSize))
		{
			mat->data = ptr;
			mat->step = stride;
            mat->refcount = (int*)cv::fastMalloc(sizeof(int));
			memoryUsage += mat->step*rows;
			Increment(ptr, mat->step*rows);
			BOOST_LOG_TRIVIAL(debug) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
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
		throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
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
			if(std::get<2>(*itr) == sizeNeeded)
			{
				mat->data = std::get<0>(*itr);
				mat->step = stride;
                mat->refcount = (int*)cv::fastMalloc(sizeof(int));
				deallocateList.erase(itr);
				memoryUsage += mat->step*rows;
				Increment(mat->data, mat->step*rows);
				BOOST_LOG_TRIVIAL(debug) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. total usage: " << memoryUsage / (1024 * 1024) << " MB";
				return true;
			}
		}
		if (rows > 1 && cols > 1)
		{
			CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
			memoryUsage += mat->step*rows;
			Increment(mat->data, mat->step*rows);
			BOOST_LOG_TRIVIAL(info) << "[GPU] Allocating block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
		}
		else
		{
			CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
			memoryUsage += elemSize*cols*rows;
			Increment(mat->data, elemSize*cols*rows);
			BOOST_LOG_TRIVIAL(info) << "[GPU] Allocating block of size (" << rows << "," << cols << ") " << cols * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
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
		BOOST_LOG_TRIVIAL(debug) << "[GPU] Releasing mat of size (" << mat->rows << "," << mat->cols << ") " << (mat->dataend - mat->datastart)/(1024*1024) << " MB to the memory pool";
		deallocateList.push_back(std::make_tuple(mat->data, clock(), size_t(mat->rows*mat->step)));
		cv::fastFree(mat->refcount);
		clear();
	}
	void DelayedDeallocator::clear()
	{
		auto time = clock();
		for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
		{
			if((time - std::get<1>(*itr)) > deallocateDelay)
			{
				memoryUsage -= std::get<2>(*itr);
				BOOST_LOG_TRIVIAL(info) << "[GPU] Deallocating block of size " << std::get<2>(*itr) /(1024*1024) << "MB. Which was stale for " << time - std::get<1>(*itr) << " ms";
				CV_CUDEV_SAFE_CALL(cudaFree(std::get<0>(*itr)));
				itr = deallocateList.erase(itr);
			}
		}
	}
	CombinedAllocator* CombinedAllocator::Instance(size_t initial_pool_size, size_t threshold_level)
	{
		static CombinedAllocator* inst = nullptr;
		if (inst == nullptr)
		{
			inst = new CombinedAllocator(initial_pool_size, threshold_level);
		}
		return inst;
	}

	CombinedAllocator::CombinedAllocator(size_t initial_pool_size, size_t threshold_level) :
		initialBlockSize_(initial_pool_size), DelayedDeallocator(), _threshold_level(threshold_level)
	{
		blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(initialBlockSize_)));
	}
	bool CombinedAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
	{
		if (rows*cols*elemSize < _threshold_level)
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
					mat->refcount = (int*)cv::fastMalloc(sizeof(int));
					memoryUsage += mat->step*rows;
					BOOST_LOG_TRIVIAL(debug) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
					Increment(ptr, mat->step*rows);
					return true;
				}
			}
			// If we get to this point, then no memory was found, need to allocate new memory
			blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(initialBlockSize_ / 2, sizeNeeded))));
			BOOST_LOG_TRIVIAL(warning) << "[GPU] Expanding memory pool by " << std::max(initialBlockSize_ / 2, sizeNeeded) / (1024 * 1024) << " MB";
			if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, elemSize))
			{
				mat->data = ptr;
				mat->step = stride;
				mat->refcount = (int*)cv::fastMalloc(sizeof(int));
				memoryUsage += mat->step*rows;
				Increment(ptr, mat->step*rows);
				BOOST_LOG_TRIVIAL(debug) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
				return true;
			}
		}
		return DelayedDeallocator::allocate(mat, rows, cols, elemSize);
	}
	void CombinedAllocator::free(cv::cuda::GpuMat* mat)
	{
		std::lock_guard<std::recursive_mutex> lock(mtx);
		for (auto itr : blocks)
		{
			if (mat->data > itr->begin && mat->data < itr->end)
			{
				if (itr->deAllocate(mat->data))
				{
					cv::fastFree(mat->refcount);
					return;
				}
			}
		}
		DelayedDeallocator::free(mat);
	}
}

