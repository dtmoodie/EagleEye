#include "SyncedMemory.h"

using namespace EagleLib;
SyncedMemory::SyncedMemory()
{
	sync_flags.resize(1, SYNCED);
}

SyncedMemory::SyncedMemory(cv::MatAllocator* cpu_allocator, cv::cuda::GpuMat::Allocator* gpu_allocator):
	h_data(1, cv::Mat()), d_data(1, cv::cuda::GpuMat(gpu_allocator)), sync_flags(1, SYNCED)
{
	h_data[0].allocator = cpu_allocator;
}

const cv::Mat&				
SyncedMemory::GetMat(cv::cuda::Stream& stream, int index)
{
	if (sync_flags[index] == DEVICE_UPDATED)
		d_data[index].download(h_data[index], stream);
	return h_data[index];
}

cv::Mat&					
SyncedMemory::GetMatMutable(cv::cuda::Stream& stream, int index)
{
	if (sync_flags[index] == DEVICE_UPDATED)
		d_data[index].download(h_data[index], stream);
	sync_flags[index] = HOST_UPDATED;
	return h_data[index];
}

const cv::cuda::GpuMat&		
SyncedMemory::GetGpuMat(cv::cuda::Stream& stream, int index)
{
	if (sync_flags[index] == HOST_UPDATED)
		d_data[index].upload(h_data[index], stream);
	return d_data[index];
}

cv::cuda::GpuMat&			
SyncedMemory::GetGpuMatMutable(cv::cuda::Stream& stream, int index)
{
	if (sync_flags[index] == HOST_UPDATED)
		d_data[index].upload(h_data[index], stream);
	sync_flags[index] = DEVICE_UPDATED;
	return d_data[index];
}

const std::vector<cv::Mat>&
SyncedMemory::GetMatVec(cv::cuda::Stream& stream)
{
	for (int i = 0; i < sync_flags.size(); ++i)
	{
		if (sync_flags[i] == DEVICE_UPDATED)
			d_data[i].download(h_data[i], stream);
	}
	return h_data;
}

std::vector<cv::Mat>&
SyncedMemory::GetMatVecMutable(cv::cuda::Stream& stream)
{
	for (int i = 0; i < sync_flags.size(); ++i)
	{
		if (sync_flags[i] == DEVICE_UPDATED)
			d_data[i].download(h_data[i], stream);
		sync_flags[i] = HOST_UPDATED;
	}
	
	return h_data;
}

const std::vector<cv::cuda::GpuMat>&
SyncedMemory::GetGpuMatVec(cv::cuda::Stream& stream)
{
	for (int i = 0; i < sync_flags.size(); ++i)
	{
		if (sync_flags[i] == HOST_UPDATED)
			d_data[i].upload(h_data[i], stream);
	}
	return d_data;
}

std::vector<cv::cuda::GpuMat>&
SyncedMemory::GetGpuMatVecMutable(cv::cuda::Stream& stream)
{
	for (int i = 0; i < sync_flags.size(); ++i)
	{
		if (sync_flags[i] == HOST_UPDATED)
			d_data[i].upload(h_data[i], stream);
		sync_flags[i] = DEVICE_UPDATED;
	}
	return d_data;
}