#include "EagleLib/utilities/MemoryBlock.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <opencv2/cudev/common.hpp>
#include <cuda_runtime.h>
using namespace EagleLib;

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

void GPUMemory::_allocate(unsigned char** ptr, size_t size)
{
    CV_CUDEV_SAFE_CALL(cudaMalloc(ptr, size));
}
void GPUMemory::_deallocate(unsigned char* ptr)
{
    CV_CUDEV_SAFE_CALL(cudaFree(ptr));
}

void CPUMemory::_allocate(unsigned char** ptr, size_t size)
{
    CV_CUDEV_SAFE_CALL(cudaMallocHost(ptr, size));
}

void CPUMemory::_deallocate(unsigned char* ptr)
{
    CV_CUDEV_SAFE_CALL(cudaFreeHost(ptr));
}

template<class XPU>
MemoryBlock<XPU>::MemoryBlock(size_t size_):
    size(size_)
{
    XPU::_allocate(&begin, size);
    end = begin + size;
}

template<class XPU>
MemoryBlock<XPU>::~MemoryBlock()
{
    XPU::_deallocate(begin);
}

template<class XPU>
unsigned char* MemoryBlock<XPU>::allocate(size_t size_, size_t elemSize_)
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
                if (static_cast<size_t>(itr.first - prevEnd + alignment) >= size_)
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
        if (static_cast<size_t>(end - prevEnd + alignment) >= size_)
        {
            candidates.push_back(std::make_pair(size_t(end - prevEnd + alignment), prevEnd + alignment));
        }
    }
    // Find the smallest chunk of memory that fits our requirement, helps reduce fragmentation.
    auto min = std::min_element(candidates.begin(), candidates.end(),
                    [](const std::pair<size_t, unsigned char*>& first, const std::pair<size_t, unsigned char*>& second)
                    {
                        return first.first < second.first;
                    });

    if (min != candidates.end() && min->first > size_)
    {
        allocatedBlocks[min->second] = (unsigned char*)(min->second + size_);
        return min->second;
    }
    return nullptr;
}

template<class XPU>
bool MemoryBlock<XPU>::deAllocate(unsigned char* ptr)
{
    if (ptr < begin || ptr > end)
        return false;
    auto itr = allocatedBlocks.find(ptr);
    if (itr != allocatedBlocks.end())
    {
        allocatedBlocks.erase(itr);
        return true;
    }
    return true;
}

template<class XPU>
unsigned char* MemoryBlock<XPU>::Begin() const
{
    return begin;
}

template<class XPU>
unsigned char* MemoryBlock<XPU>::End() const
{
    return end;
}

template<class XPU>
size_t MemoryBlock<XPU>::Size() const
{
    return size;
}

template class MemoryBlock<CPUMemory>;
template class MemoryBlock<GPUMemory>;
