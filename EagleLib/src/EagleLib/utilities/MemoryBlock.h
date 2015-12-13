#pragma once
#include <map>
#include <functional>
#include "EagleLib/Defs.hpp"
namespace EagleLib
{
	int alignmentOffset(unsigned char* ptr, int elemSize);
	unsigned char*  alignMemory(unsigned char* ptr, int elemSize);
	class EAGLE_EXPORTS MemoryBlock
	{
	public:
		MemoryBlock(size_t size_);
		~MemoryBlock();
		
		virtual unsigned char* allocate(size_t size_, size_t elemSize_);
		virtual bool deAllocate(unsigned char* ptr);
		unsigned char* begin;
		unsigned char* end;
		size_t size;
	protected:
		virtual void _allocate(unsigned char** data, size_t size) = 0;
		virtual void _deallocate(unsigned char* data) = 0;
		
		std::map<unsigned char*, unsigned char*> allocatedBlocks;
	};
	class EAGLE_EXPORTS GpuMemoryBlock : public MemoryBlock
	{
	public:
		GpuMemoryBlock(size_t initialSize);
		~GpuMemoryBlock();
	private:
		virtual void _allocate(unsigned char** ptr, size_t size);
		virtual void _deallocate(unsigned char* ptr);
	};
	class EAGLE_EXPORTS CpuMemoryBlock : public MemoryBlock
	{
	public:
		CpuMemoryBlock(size_t initialSize);
		~CpuMemoryBlock();
	private:
		virtual void _allocate(unsigned char** ptr, size_t size);
		virtual void _deallocate(unsigned char* ptr);
	};
}