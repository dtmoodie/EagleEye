#include "CpuMatAllocators.h"
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/core_c.h>
#include <boost/log/trivial.hpp>
#include "MemoryBlock.h"
using namespace EagleLib;



CpuDelayedDeallocationPool::CpuDelayedDeallocationPool(size_t initial_pool_size, size_t threshold_level):
	_threshold_level(threshold_level), _initial_block_size(initial_pool_size)
{
	blocks.push_back(std::shared_ptr<CpuMemoryBlock>(new CpuMemoryBlock(initial_pool_size)));
    deallocation_delay = 100;
	total_usage = 0;
}
CpuDelayedDeallocationPool::~CpuDelayedDeallocationPool()
{
	cleanup(true);
}

CpuDelayedDeallocationPool* CpuDelayedDeallocationPool::instance(size_t initial_pool_size, size_t threshold_level)
{
	static CpuDelayedDeallocationPool* g_instance = nullptr;
	if (g_instance == nullptr)
	{
		g_instance = new CpuDelayedDeallocationPool(initial_pool_size, threshold_level);
	}
	return g_instance;
}

void CpuDelayedDeallocationPool::allocate(void** ptr, size_t total)
{
	auto inst = instance();
	std::lock_guard<std::recursive_timed_mutex> lock(inst->deallocate_pool_mutex);
	if (total < inst->_threshold_level)
	{
		unsigned char* _ptr;
		for (auto& block : inst->blocks)
		{
			_ptr = block->allocate(total, 1);
			if (_ptr)
			{
				*ptr = _ptr;
				return;
			}
		}
		inst->blocks.push_back(
			std::shared_ptr<CpuMemoryBlock>(
				new CpuMemoryBlock(std::max(inst->_initial_block_size / 2, total))));

		if (_ptr = (*inst->blocks.rbegin())->allocate(total, 1))
		{
			return;
		}
		throw cv::Exception(-1, "Failed to allocate sufficient page locked memory", __FUNCTION__, __FILE__, __LINE__);
	}
	for (auto itr = inst->deallocate_pool.begin(); itr != inst->deallocate_pool.end(); ++itr)
	{
		if(std::get<2>(*itr) == total)
		{
			*ptr = std::get<0>(*itr);
			inst->deallocate_pool.erase(itr);
            BOOST_LOG_TRIVIAL(trace) << "[CPU] Reusing memory block of size " << total / (1024 * 1024) << " MB. Total usage: " << inst->total_usage /(1024*1024) << " MB";
			return;
		}
	}
	inst->total_usage += total;
    BOOST_LOG_TRIVIAL(info) << "[CPU] Allocating block of size " << total / (1024 * 1024) << " MB. Total usage: " << inst->total_usage / (1024 * 1024) << " MB";
	cudaSafeCall(cudaMallocHost(ptr, total));
}

void CpuDelayedDeallocationPool::deallocate(void* ptr, size_t total)
{
	auto inst = instance();
    std::lock_guard<std::recursive_timed_mutex> lock(inst->deallocate_pool_mutex);
	for (auto itr : inst->blocks)
	{
		if (ptr > itr->begin && ptr < itr->end)
		{
			if (itr->deAllocate((unsigned char*)ptr))
			{
				return;
			}

		}
	}
	inst->deallocate_pool.push_back(std::make_tuple((unsigned char*)ptr, clock(), total));
    inst->cleanup();
}
void CpuDelayedDeallocationPool::cleanup(bool force)
{
	auto time = clock();
	if (force)
		time = 0;
	for (auto itr = deallocate_pool.begin(); itr != deallocate_pool.end(); ++itr)
	{
		//if ((time - itr->second.first) > deallocation_delay)
		if((time - std::get<1>(*itr)) > deallocation_delay)
		{
			//total_usage -= itr->second.second;
			total_usage -= std::get<2>(*itr);
            BOOST_LOG_TRIVIAL(info) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
				<< " MB. Which was stale for " << time - std::get<1>(*itr)
				<< " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
			cudaFreeHost((void*)std::get<0>(*itr));
			itr = deallocate_pool.erase(itr);
		}
	}
}
EagleLib::CpuPinnedAllocator* EagleLib::CpuPinnedAllocator::instance()
{
	static EagleLib::CpuPinnedAllocator inst;
	return &inst;
}
cv::UMatData* EagleLib::CpuPinnedAllocator::allocate(int dims, const int* sizes, int type,
	void* data0, size_t* step,
	int /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const
{
	size_t total = CV_ELEM_SIZE(type);
	for (int i = dims - 1; i >= 0; i--)
	{
		if (step)
		{
			if (data0 && step[i] != CV_AUTOSTEP)
			{
				CV_Assert(total <= step[i]);
				total = step[i];
			}
			else
			{
				step[i] = total;
			}
		}

		total *= sizes[i];
	}

	cv::UMatData* u = new cv::UMatData(this);
	u->size = total;

	if (data0)
	{
		u->data = u->origdata = static_cast<uchar*>(data0);
		u->flags |= cv::UMatData::USER_ALLOCATED;
	}
	else
	{
		void* ptr = 0;
		CpuDelayedDeallocationPool::allocate(&ptr, total);
		//cudaSafeCall(cudaMallocHost(&ptr, total));
		//cudaSafeCall(cudaHostAlloc(&ptr, total));

		u->data = u->origdata = static_cast<uchar*>(ptr);
	}

	return u;
}

bool EagleLib::CpuPinnedAllocator::allocate(cv::UMatData* u, int /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/) const
{
	return (u != NULL);
}

void EagleLib::CpuPinnedAllocator::deallocate(cv::UMatData* u) const
{
	if (!u)
		return;

	CV_Assert(u->urefcount >= 0);
	CV_Assert(u->refcount >= 0);

	if (u->refcount == 0)
	{
		if (!(u->flags & cv::UMatData::USER_ALLOCATED))
		{
			//cudaFreeHost(u->origdata);
			CpuDelayedDeallocationPool::deallocate(u->origdata, u->size);
			u->origdata = 0;
		}

		delete u;
	}
}