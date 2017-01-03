#include "EagleLib/utilities/CpuMatAllocators.h"
#include "MetaObject/Detail/MemoryBlock.h"
#include <cuda_runtime_api.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/core_c.h>
#include <MetaObject/Logging/Log.hpp>


using namespace EagleLib;



CpuDelayedDeallocationPool::CpuDelayedDeallocationPool(size_t initial_pool_size, size_t threshold_level):
    _threshold_level(threshold_level), _initial_block_size(initial_pool_size)
{
    blocks.push_back(std::shared_ptr<mo::CpuMemoryBlock>(new mo::CpuMemoryBlock(initial_pool_size)));
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

void CpuDelayedDeallocationPool::allocate(void** ptr, size_t total, size_t elemSize)
{
    std::lock_guard<std::recursive_timed_mutex> lock(deallocate_pool_mutex);
    LOG(trace) << "Requesting allocation of " << total << " bytes";
    *ptr = nullptr;
    if (total < _threshold_level && false)
    {
        LOG(trace) << "Requesting allocation is less than threshold, using block memory allocation";
        int index = 0;
        unsigned char* _ptr;
        for (auto& block : blocks)
        {
            _ptr = block->allocate(total, elemSize);
            if (_ptr)
            {
                *ptr = _ptr;
                LOG(trace) << "Allocating " << total << " bytes from pre-allocated memory block number " << index << " at address: " << (void*)_ptr;
                return;
            }
            ++index;
        }
        LOG(trace) << "Creating new block of page locked memory for allocation.";
        blocks.push_back(
            std::shared_ptr<mo::CpuMemoryBlock>(
                new mo::CpuMemoryBlock(std::max(_initial_block_size / 2, total))));
        _ptr = (*blocks.rbegin())->allocate(total, elemSize);
        if (_ptr)
        {
            LOG(debug) << "Allocating " << total << " bytes from newly created memory block at address: " << (void*)_ptr;
            *ptr = _ptr;
            return;
        }
        throw cv::Exception(-1, "Failed to allocate sufficient page locked memory", __FUNCTION__, __FILE__, __LINE__);
    }
    LOG(trace) << "Requested allocation is greater than threshold, using lazy deallocation pool";
    for (auto itr = deallocate_pool.begin(); itr != deallocate_pool.end(); ++itr)
    {
        if(std::get<2>(*itr) == total)
        {
            *ptr = std::get<0>(*itr);
            deallocate_pool.erase(itr);
            LOG(trace) << "[CPU] Reusing memory block of size " << total / (1024 * 1024) << " MB. Total usage: " << total_usage /(1024*1024) << " MB";
            return;
        }
    }
    total_usage += total;
    LOG(trace) << "[CPU] Allocating block of size " << total / (1024 * 1024) << " MB. Total usage: " << total_usage / (1024 * 1024) << " MB";
    cudaSafeCall(cudaMallocHost(ptr, total));
}

void CpuDelayedDeallocationPool::deallocate(void* ptr, size_t total)
{
    std::lock_guard<std::recursive_timed_mutex> lock(deallocate_pool_mutex);
    /*
    for (auto itr : blocks)
    {
        if (ptr > itr->begin && ptr < itr->end)
        {
            BOOST_LOG(trace) << "Releasing memory block of size " << total << " at address: " << ptr;
            if (itr->deAllocate((unsigned char*)ptr))
            {
                return;
            }
        }
    }
    */
    LOG(trace) << "Releasing " << total / (1024 * 1024) << " MB to lazy deallocation pool";
    deallocate_pool.push_back(std::make_tuple((unsigned char*)ptr, clock(), total));
    cleanup();
}
void CpuDelayedDeallocationPool::cleanup(bool force)
{
    auto time = clock();
    if (force)
        time = 0;
    for (auto itr = deallocate_pool.begin(); itr != deallocate_pool.end();)
    {
        if((time - std::get<1>(*itr)) > deallocation_delay)
        {
            total_usage -= std::get<2>(*itr);
            LOG(trace) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
                << " MB. Which was stale for " << time - std::get<1>(*itr)
                << " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
            cudaFreeHost((void*)std::get<0>(*itr));
            itr = deallocate_pool.erase(itr);
        }else
        {
            ++itr;
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
        CpuDelayedDeallocationPool::instance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

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
            CpuDelayedDeallocationPool::instance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}
