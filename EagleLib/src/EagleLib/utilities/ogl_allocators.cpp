#include "ogl_allocators.h"

using namespace EagleLib;
ogl_allocator* ogl_allocator::instance(size_t initial_pool_size, size_t threshold_level)
{
    static ogl_allocator* inst = nullptr;
    if(inst == nullptr)
    {
        inst = new ogl_allocator(initial_pool_size, threshold_level);
    }
    return inst;
}
ogl_allocator::ogl_allocator(size_t initial_pool_size, size_t threshold_level):
    CombinedAllocator(initial_pool_size, threshold_level), BlockMemoryAllocator(initial_pool_size)
{

}
bool ogl_allocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    return CombinedAllocator::allocate(mat, rows, cols, elemSize);
}

void ogl_allocator::free(cv::cuda::GpuMat* mat)
{
    CombinedAllocator::free(mat);
}


// Creates a GpuMat from an opengl buffer
cv::cuda::GpuMat ogl_allocator::createMat(int rows, int cols, int type, cv::cuda::Stream& stream)
{
	for (auto itr = unused_ogl_buffers.begin(); itr !=  unused_ogl_buffers.end(); ++itr)
	{
		if(itr->rows() == rows && itr->cols() == cols && itr->type() == type)
        {
            auto mat = itr->mapDevice(stream);
            mapped_buffers[mat.data] = *itr;
            unused_ogl_buffers.erase(itr);
            return mat;
        }
	}
	cv::ogl::Buffer buffer(rows, cols, type, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
    auto mat = buffer.mapDevice(stream);
    mapped_buffers[mat.data] = buffer;
    return mat;
}
cv::ogl::Buffer ogl_allocator::mapBuffer(cv::cuda::GpuMat& mat, cv::cuda::Stream& stream)
{
    auto itr = mapped_buffers.find(mat.data);
    if(itr != mapped_buffers.end())
    {
        itr->second.unmapDevice(stream);
        used_ogl_buffers.push_back(itr->second);
        return itr->second;
    }
}