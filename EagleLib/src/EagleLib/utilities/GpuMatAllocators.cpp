#include "EagleLib/utilities/GpuMatAllocators.h"
#include "MetaObject/Detail/MemoryBlock.h"
#include "EagleLib/logger.hpp"

#include <boost/log/trivial.hpp>
#include <boost/thread.hpp>
#include <boost/thread/tss.hpp>
#include <opencv2/cudev/common.hpp>
#include <cuda_runtime.h>
using namespace mo;
namespace EagleLib
{
    PitchedAllocator::PitchedAllocator()
    {
        textureAlignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
        memoryUsage = 0;
        SetScope("Default");
    }

    void PitchedAllocator::SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride)
    {
        if (rows == 1 || cols == 1)
        {
            stride = cols*elemSize;
        }
        else
        {
            if((cols*elemSize % textureAlignment) == 0)
                stride = cols*elemSize;
            else
                stride = cols*elemSize + textureAlignment - (cols*elemSize % textureAlignment);
        }
        sizeNeeded = stride*rows;
    }

    void PitchedAllocator::SetScope(const std::string& name)
    {
        auto id = boost::this_thread::get_id();
        auto itr = scopedAllocationSize.find(name);
        if (itr == scopedAllocationSize.end())
        {
            scopedAllocationSize[name] = 0;
            currentScopeName[id] = name; 
        }
        currentScopeName[id] = name;
    }

    void PitchedAllocator::Increment(unsigned char* ptr, size_t size)
    {
        auto id = boost::this_thread::get_id();
        scopeOwnership[ptr] = currentScopeName[id];
        scopedAllocationSize[currentScopeName[id]] += size;
    }
    void PitchedAllocator::Decrement(unsigned char* ptr, size_t size)
    {
        auto itr = scopeOwnership.find(ptr);
        if (itr != scopeOwnership.end())
        {
            scopedAllocationSize[itr->second] -= size;
        }        
    }

    // =====================================================================
    // BlockMemoryAllocator
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
                LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
                Increment(ptr, mat->step*rows);
                return true;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(initialBlockSize_ / 2, sizeNeeded))));
        LOG(trace) << "[GPU] Expanding memory pool by " <<  std::max(initialBlockSize_ / 2, sizeNeeded) / (1024 * 1024) << " MB";
        if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, elemSize))
        {
            mat->data = ptr;
            mat->step = stride;
            mat->refcount = (int*)cv::fastMalloc(sizeof(int));
            memoryUsage += mat->step*rows;
            Increment(ptr, mat->step*rows);
            LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
            return true;
        }
        return false;
    }
    unsigned char* BlockMemoryAllocator::allocate(size_t sizeNeeded)
    {

        unsigned char* ptr;
        for (auto itr : blocks)
        {
            ptr = itr->allocate(sizeNeeded, 1);
            if (ptr)
            {
                memoryUsage += sizeNeeded;
                Increment(ptr, sizeNeeded);
                return ptr;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(initialBlockSize_ / 2, sizeNeeded))));
        LOG(trace) << "[GPU] Expanding memory pool by " << std::max(initialBlockSize_ / 2, sizeNeeded) / (1024 * 1024) << " MB";
        if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, 1))
        {
            memoryUsage += sizeNeeded;
            Increment(ptr, sizeNeeded);
            return ptr;
        }
        return nullptr;
    }
    void BlockMemoryAllocator::free(unsigned char* ptr)
    {

        for (auto itr : blocks)
        {
            if (itr->deAllocate(ptr))
            {
                return;
            }
        }
        throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
    }
    void BlockMemoryAllocator::free(cv::cuda::GpuMat* mat)
    {
        for (auto itr : blocks)
        {
            if (itr->deAllocate(mat->data))
            {
                cv::fastFree(mat->refcount);
                Decrement(mat->data, mat->step*mat->rows);
                memoryUsage -= mat->step*mat->rows;
                return;
            }
        }
        throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
    }

    bool BlockMemoryAllocator::free_impl(cv::cuda::GpuMat* mat)
    {
        for (auto itr : blocks)
        {
            if (itr->deAllocate(mat->data))
            {
                cv::fastFree(mat->refcount);
                Decrement(mat->data, mat->step*mat->rows);
                memoryUsage -= mat->step*mat->rows;
                return true;
            }
        }
        return false;
    }

    // =====================================================================
    // DelayedDeallocator
    DelayedDeallocator::DelayedDeallocator() :
        PitchedAllocator()
    {
        deallocateDelay = 5000;
    }
    
    bool DelayedDeallocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        // First check for anything of the correct size

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
                LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. total usage: " << memoryUsage / (1024 * 1024) << " MB";
                return true;
            }
        }
        if (rows > 1 && cols > 1)
        {
            CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
            memoryUsage += mat->step*rows;
            Increment(mat->data, mat->step*rows);
            LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") " << mat->step * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
        }
        else
        {
            CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
            memoryUsage += elemSize*cols*rows;
            Increment(mat->data, elemSize*cols*rows);
            LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") " << cols * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
            mat->step = elemSize * cols;
        }
        mat->refcount = (int*)cv::fastMalloc(sizeof(int));
        return true;
    }
    unsigned char* DelayedDeallocator::allocate(size_t sizeNeeded)
    {
        unsigned char* ptr = nullptr;
        for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
        {
            if (std::get<2>(*itr) == sizeNeeded)
            {
                ptr = std::get<0>(*itr);
                deallocateList.erase(itr);
                memoryUsage += sizeNeeded;
                Increment(ptr, sizeNeeded);
                current_allocations[ptr] = sizeNeeded;
                return ptr;
            }
        }
        CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, sizeNeeded));
        memoryUsage += sizeNeeded;
        Increment(ptr, sizeNeeded);
        current_allocations[ptr] = sizeNeeded;
        return ptr;
    }
    void DelayedDeallocator::free(unsigned char* ptr)
    {

        //Decrement(ptr, mat->step*mat->rows);
        //scopedAllocationSize[scopeOwnership[ptr]] -= mat->rows*mat->step;
        //memoryUsage -= mat->rows*mat->step;
        //BOOST_LOG(trace) << "[GPU] Releasing mat of size (" << mat->rows << "," << mat->cols << ") " << (mat->dataend - mat->datastart) / (1024 * 1024) << " MB to the memory pool";
        auto itr = current_allocations.find(ptr);
        if(itr != current_allocations.end())
        {
            current_allocations.erase(itr);
            deallocateList.push_back(std::make_tuple(ptr, clock(), current_allocations[ptr]));
        }      
        
        clear();
    }
    void DelayedDeallocator::free(cv::cuda::GpuMat* mat)
    {

        Decrement(mat->data, mat->step*mat->rows);
        scopedAllocationSize[scopeOwnership[mat->data]] -= mat->rows*mat->step;
        memoryUsage -= mat->rows*mat->step;
        LOG(trace) << "[GPU] Releasing mat of size (" << mat->rows << "," << mat->cols << ") " << (mat->dataend - mat->datastart)/(1024*1024) << " MB to the memory pool";
        deallocateList.push_back(std::make_tuple(mat->data, clock(), mat->dataend - mat->datastart));
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
                LOG(trace) << "[GPU] Deallocating block of size " << std::get<2>(*itr) /(1024*1024) << "MB. Which was stale for " << time - std::get<1>(*itr) << " ms";
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
        BlockMemoryAllocator(initial_pool_size),
        initialBlockSize_(initial_pool_size), DelayedDeallocator(), _threshold_level(threshold_level)
    {
        blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(initialBlockSize_)));
    }
    bool CombinedAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        if (rows*cols*elemSize < _threshold_level)
        {

            size_t sizeNeeded, stride;
            SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
            return BlockMemoryAllocator::allocate(mat, rows, cols, elemSize);
        }
        
        return DelayedDeallocator::allocate(mat, rows, cols, elemSize);
    }
    unsigned char* CombinedAllocator::allocate(size_t num_bytes)
    {
        return BlockMemoryAllocator::allocate(num_bytes);
    }
    void CombinedAllocator::free(unsigned char* ptr)
    {
        return BlockMemoryAllocator::free(ptr);
    }
    void CombinedAllocator::free(cv::cuda::GpuMat* mat)
    {

        if(!BlockMemoryAllocator::free_impl(mat))
            DelayedDeallocator::free(mat);
        
    }
}

