#pragma once
#include <opencv2/core/mat.hpp>
#include "../Defs.hpp"
#include <map>
#include <list>
#include <tuple>
#include <mutex>
#include <time.h>
namespace EagleLib
{
	class EAGLE_EXPORTS CpuDelayedDeallocationPool
	{
	public:
        CpuDelayedDeallocationPool();
		~CpuDelayedDeallocationPool();
		static CpuDelayedDeallocationPool* instance();
		static void allocate(void** ptr, size_t total);
		static void deallocate(void* ptr, size_t total);
        size_t deallocation_delay;
		size_t total_usage;
	private:
		void cleanup(bool force = false);
		//std::map<unsigned char*, std::pair<clock_t, size_t>> deallocate_pool;
		std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_pool;
		std::recursive_timed_mutex deallocate_pool_mutex;
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