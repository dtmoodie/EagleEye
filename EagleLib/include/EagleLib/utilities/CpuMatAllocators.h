#pragma once
#include "MetaObject/Detail/MemoryBlock.h"
#include <opencv2/core/mat.hpp>
#include "EagleLib/Detail/Export.hpp"
#include <map>
#include <list>
#include <tuple>
#include <mutex>
#include <time.h>
#include <memory>
namespace EagleLib
{

    class EAGLE_EXPORTS CpuDelayedDeallocationPool
    {
    public:
        CpuDelayedDeallocationPool(size_t initial_pool_size, size_t threshold_level);
        ~CpuDelayedDeallocationPool();
        static CpuDelayedDeallocationPool* instance(size_t initial_pool_size = 10000000, size_t threshold_level = 1000000);

        void allocate(void** ptr, size_t total, size_t elemSize);
        void deallocate(void* ptr, size_t total);
        size_t deallocation_delay;
        size_t total_usage;
        size_t _threshold_level;
    protected:
        void cleanup(bool force = false);
        std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_pool;
        std::recursive_timed_mutex deallocate_pool_mutex;
        
        size_t _initial_block_size;
        std::list<std::shared_ptr<mo::CpuMemoryBlock>> blocks;
    };

    class EAGLE_EXPORTS CpuPinnedAllocator : public cv::MatAllocator
    {
    public:
        static CpuPinnedAllocator* instance();
        virtual cv::UMatData* allocate(int dims, const int* sizes, int type,
            void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
        virtual bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
        virtual void deallocate(cv::UMatData* data) const;
    };

}
