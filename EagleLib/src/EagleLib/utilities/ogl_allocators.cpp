#include "ogl_allocators.h"

using namespace EagleLib;
bool ogl_allocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{

}

void ogl_allocator::free(cv::cuda::GpuMat* mat)
{

}


// Creates a GpuMat from an opengl buffer
cv::cuda::GpuMat ogl_allocator::createMat(int rows, int cols, int type, cv::cuda::Stream& stream)
{
	for (auto& itr : unused_ogl_buffers)
	{
		
	}
	cv::ogl::Buffer buffer(rows, cols, type, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
}
cv::ogl::Buffer ogl_allocator::mapBuffer(cv::cuda::GpuMat& mat, cv::cuda::Stream& stream)
{

}