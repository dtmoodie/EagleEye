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
    auto id = boost::this_thread::get_id();
    auto scope_name = currentScopeName[id];
    auto scope = std::make_tuple(scope_name, rows, cols, mat->type(), id, false);
    auto itr = std::find_if(buffer_request_locations.begin(), buffer_request_locations.end(), [&scope](std::tuple<std::string, int, int, int, boost::thread::id, bool>& other)->bool
    {
        return std::get<0>(other) == std::get<0>(scope) && 
               std::get<1>(other) == std::get<1>(scope) && 
               std::get<2>(other) == std::get<2>(scope) && 
               std::get<3>(other) == std::get<3>(scope) && 
               std::get<4>(other) == std::get<4>(scope);
    });

    if(itr == buffer_request_locations.end())
    {
        buffer_request_locations.push_back(scope);
    }else
    {
        if(std::get<5>(*itr))
        {
            // Allocate from opengl buffer
            for(auto itr = unused_ogl_buffers.begin(); itr != unused_ogl_buffers.end(); ++itr)
            {
                if(itr->rows() == rows && itr->cols() == cols && itr->type() == mat->type())
                {
                    *mat = itr->mapDevice();
                    mat->allocator = this;
                    used_ogl_buffers.push_back(*itr);
                    unused_ogl_buffers.erase(itr);
                    return true;
                }
            }
        }
    }
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
        // Matrix wasn't originally mapped from an opengl buffer, thus we need to update buffer_request_locations to indicate that those gpumats should be allocated from opengl memory
        auto scope_name = scopeOwnership.find(mat.data);
        if(scope_name != scopeOwnership.end())
        {
            auto scope_itr = std::find_if(buffer_request_locations.begin(), buffer_request_locations.end(), [&](std::tuple<std::string, int, int, int, boost::thread::id, bool>& other)->bool
            {
                   return std::get<0>(other) == scope_name->second && std::get<1>(other) == mat.rows && std::get<2>(other) == mat.cols && std::get<3>(other) == mat.type();
            });
            std::get<5>(*scope_itr) = true;
        }
        itr->second.unmapDevice(stream);
        used_ogl_buffers.push_back(itr->second);
        return itr->second;
    }
}