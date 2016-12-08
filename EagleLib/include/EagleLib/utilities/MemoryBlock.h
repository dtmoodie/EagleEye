#pragma once
#include <map>
#include <functional>
#include "EagleLib/Detail/Export.hpp"
namespace EagleLib
{
    int alignmentOffset(unsigned char* ptr, int elemSize);
    unsigned char*  alignMemory(unsigned char* ptr, int elemSize);

    class GPUMemory
    {
    protected:
        inline void _allocate(unsigned char** data, size_t size);
        inline void _deallocate(unsigned char* data);
    };

    class CPUMemory
    {
    protected:
        inline void _allocate(unsigned char** data, size_t size);
        inline void _deallocate(unsigned char* data);
    };

    template<class XPU> class EAGLE_EXPORTS MemoryBlock: public XPU
    {
    public:
        MemoryBlock(size_t size_);
        ~MemoryBlock();
        
        unsigned char* allocate(size_t size_, size_t elemSize_);
        bool deAllocate(unsigned char* ptr);
        unsigned char* Begin() const;
        unsigned char* End() const;
        size_t Size() const;
    protected:
        unsigned char* begin;
        unsigned char* end;
        size_t size;
        std::map<unsigned char*, unsigned char*> allocatedBlocks;
    };
    typedef MemoryBlock<GPUMemory> GpuMemoryBlock;
    typedef MemoryBlock<CPUMemory> CpuMemoryBlock;
}
