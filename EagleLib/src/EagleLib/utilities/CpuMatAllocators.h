#pragma once
#include <opencv2/core/mat.hpp>
#include "../Defs.hpp"
#include <map>
#include <mutex>
#include <time.h>
namespace EagleLib
{
	class EAGLE_EXPORTS CpuDelayedDeallocationPool
	{
	public:
		static CpuDelayedDeallocationPool* instance();
		static void allocate(void** ptr, size_t total);
		static void deallocate(void* ptr, size_t total);
	private:
		size_t deallocation_delay;
		void cleanup();
		std::map<unsigned char*, std::pair<clock_t, size_t>> deallocate_pool;
		std::mutex deallocate_pool_mutex;
	};

	class EAGLE_EXPORTS CpuPinnedAllocator : public cv::MatAllocator
	{
	public:
		virtual cv::UMatData* allocate(int dims, const int* sizes, int type,
			void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
		virtual bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
		virtual void deallocate(cv::UMatData* data) const;
	};
}