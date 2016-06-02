#include "SyncedMemory.h"
#ifdef HAVE_MXNET



#endif


using namespace EagleLib;
SyncedMemory::SyncedMemory()
{
	sync_flags.resize(1, SYNCED);
}
SyncedMemory::SyncedMemory(const cv::Mat& h_mat)
{
    h_data.resize(1, h_mat);
    d_data.resize(1);
    sync_flags.resize(1, HOST_UPDATED);
}
SyncedMemory::SyncedMemory(const cv::cuda::GpuMat& d_mat)
{
    h_data.resize(1);
    d_data.resize(1, d_mat);
    sync_flags.resize(1, DEVICE_UPDATED);
}
SyncedMemory::SyncedMemory(const cv::Mat& h_mat, const cv::cuda::GpuMat& d_mat)
{
    h_data.resize(1, h_mat);
    d_data.resize(1, d_mat);
    sync_flags.resize(1, SYNCED);
}

SyncedMemory::SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat):
    h_data(h_mat), d_data(d_mat)
{
    assert(h_data.size() == d_data.size());
    sync_flags.resize(h_data.size(), SYNCED);
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
	{
		d_data[index].download(h_data[index], stream);
		sync_flags[index] = SYNCED;
	}	
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
	{
		d_data[index].upload(h_data[index], stream);
		sync_flags[index] = SYNCED;
	}
	
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
void SyncedMemory::ResizeNumMats(int new_size)
{
    h_data.resize(new_size);
    d_data.resize(new_size);
    sync_flags.resize(new_size);
}

SyncedMemory SyncedMemory::clone(cv::cuda::Stream& stream)
{
    SyncedMemory output;
    output.h_data.resize(h_data.size());
    output.d_data.resize(d_data.size());
    for(int i = 0; i < h_data.size(); ++i)
    {
        output.h_data[i] = h_data[i].clone();
        d_data[i].copyTo(output.d_data[i], stream);
    }
    output.sync_flags = sync_flags;
    return output;
}
int SyncedMemory::GetNumMats() const
{
    return h_data.size();
}
bool SyncedMemory::empty() const
{
    if(h_data.size())
        return h_data[0].empty();
    return true;
}
void SyncedMemory::Synchronize()
{
	for(int i = 0; i < h_data.size(); ++i)
	{
		if(sync_flags[i] == HOST_UPDATED)
			d_data[i].upload(h_data[i]);
		else if(sync_flags[i] == DEVICE_UPDATED)
			d_data[i].download(h_data[i]);
	}
}

cv::Size SyncedMemory::GetSize(int index) const
{
	CV_Assert(index >= 0 && index < d_data.size());
	return d_data[index].size();
}

std::vector<int> SyncedMemory::GetShape() const
{
	std::vector<int> output;
	output.push_back(d_data.size());
	if(d_data.empty())
		return output;
	output.push_back(d_data[0].rows);
	output.push_back(d_data[0].cols);
	output.push_back(d_data[0].channels());
	return output;
}
int SyncedMemory::GetDepth() const
{
	CV_Assert(d_data.size());
	return d_data[0].depth();
}