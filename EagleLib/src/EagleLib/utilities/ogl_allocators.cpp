#include "EagleLib/utilities/ogl_allocators.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/SystemTable.hpp"
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <QOpenGLContext>
#include <ISimpleSerializer.h>
#include <qwindow.h>

using namespace EagleLib;
/*ogl_allocator* ogl_allocator::instance(size_t initial_pool_size, size_t threshold_level)
{
    static ogl_allocator* inst = nullptr;
    if(inst == nullptr)
    {
        inst = new ogl_allocator(initial_pool_size, threshold_level);
    }
    return inst;
}*/
void ogl_allocator::NodeInit(bool firstInit)
{
    if(!firstInit)
    {
        if(is_default_allocator)
            cv::cuda::GpuMat::setDefaultAllocator(this);
    }
}
void ogl_allocator::Serialize(ISimpleSerializer* pSerializer)
{
    IObject::Serialize(pSerializer);
    if(cv::cuda::GpuMat::defaultAllocator() == this)
        is_default_allocator = true;
    SERIALIZE(_pool);
    SERIALIZE(mapped_buffers);
    SERIALIZE(buffer_request_locations);
    SERIALIZE(_threshold_level);
    SERIALIZE(initialBlockSize_);
    SERIALIZE(ogl_allocator::blocks);
    SERIALIZE(ogl_allocator::currentScopeName);
    SERIALIZE(ogl_allocator::deallocateDelay);
    SERIALIZE(ogl_allocator::deallocateList);
    SERIALIZE(ogl_allocator::is_default_allocator);
    SERIALIZE(ogl_allocator::textureAlignment);
    SERIALIZE(ogl_allocator::scopeOwnership);
    SERIALIZE(ogl_allocator::scopedAllocationSize);
    SERIALIZE(ogl_allocator::memoryUsage);
}
//size_t initial_pool_size = 10000000, size_t threshold_level = 1000000
ogl_allocator::ogl_allocator():
    CombinedAllocator(10000000, 1000000), is_default_allocator(false)
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if(table)
        table->SetSingleton<ogl_allocator>(this);
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
            {

                /*auto ctx = QOpenGLContext::currentContext();
                if(ctx == nullptr)
                {
                    ctx = new QOpenGLContext();
                    ctx->setFormat(QSurfaceFormat());
                    auto global_ctx = QOpenGLContext::globalShareContext();
                    if(global_ctx)
                    {
                        ctx->setShareContext(global_ctx);
                    }
                    ctx->create();
                    ctx->makeCurrent(new QWindow());
                }*/
                
                std::lock_guard<std::mutex> pool_lock(EagleLib::pool::ObjectPool<cv::ogl::Buffer>::mtx);
                for(auto itr = _pool.begin(); itr != _pool.end(); ++itr)
                {
                    if((*itr)->rows() == rows && (*itr)->cols() == cols && (*itr)->type() == mat->type())
                    {
                        *mat = (*itr)->mapDevice();
                        mat->allocator = this;
                        mapped_buffers[mat->data] = *itr;
                        _pool.erase(itr);
                        return true;
                    }
                }
            }
            // Unable to allocate from one of the currently unused opengl buffers, so instead we will create a new buffer
            auto buf = new cv::ogl::Buffer(rows, cols, mat->type(), cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
            *mat = buf->mapDevice();
            mapped_buffers[mat->data] = buf;
            mat->allocator = this;
            return true;
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

    {
        std::lock_guard<std::mutex> pool_lock(EagleLib::pool::ObjectPool<cv::ogl::Buffer>::mtx);
        for (auto itr = _pool.begin(); itr != _pool.end(); ++itr)
        {
            if ((*itr)->rows() == rows && (*itr)->cols() == cols && (*itr)->type() == type)
            {
                auto mat = (*itr)->mapDevice(stream);
                mat.allocator = this;
                mapped_buffers[mat.data] = *itr;
                _pool.erase(itr);
                return mat;
            }
        }
    }
    cv::ogl::Buffer* buffer = new cv::ogl::Buffer(rows, cols, type, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
    auto mat = buffer->mapDevice(stream);
    mapped_buffers[mat.data] = buffer;
    return mat;
}

EagleLib::pool::Ptr<cv::ogl::Buffer> ogl_allocator::get_ogl_buffer(const cv::cuda::GpuMat& mat, cv::cuda::Stream& stream)
{

    auto itr = mapped_buffers.find(mat.data);
    if(itr != mapped_buffers.end())
    {
        itr->second->unmapDevice(stream);
        return pool::Ptr<cv::ogl::Buffer>(itr->second, this);
    }else
    {
        // Matrix wasn't originally mapped from an opengl buffer, thus we need to update buffer_request_locations to indicate that those gpumats should be allocated from opengl memory
        // List it as a GpuMat that needs to be created from an opengl buffer at next chance
        auto scope_name = scopeOwnership.find(mat.data);
        if (scope_name != scopeOwnership.end())
        {
            auto scope_itr = std::find_if(buffer_request_locations.begin(), buffer_request_locations.end(), [&](std::tuple<std::string, int, int, int, boost::thread::id, bool>& other)->bool
            {
                return std::get<0>(other) == scope_name->second && std::get<1>(other) == mat.rows && std::get<2>(other) == mat.cols && std::get<3>(other) == mat.type();
            });
            std::get<5>(*scope_itr) = true;
        }
        // Find any unused opengl buffers of the correct size
        {
            std::lock_guard<std::mutex> pool_lock(EagleLib::pool::ObjectPool<cv::ogl::Buffer>::mtx);
            for(auto buf: _pool)
            {
                if(buf->rows() == mat.rows && buf->cols() == mat.cols && buf->type() == mat.type())
                {
                    buf->copyFrom(mat, stream, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
                    return pool::Ptr<cv::ogl::Buffer>(buf, this);
                }
            }
        }
        // Need to create a new opengl buffer
        cv::ogl::Buffer* buf = new cv::ogl::Buffer;
        buf->copyFrom(mat, stream, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
        //used_ogl_buffers.push_back(buf);
        return pool::Ptr<cv::ogl::Buffer>(buf, this);
    }
}
EagleLib::pool::Ptr<cv::ogl::Buffer> ogl_allocator::get_ogl_buffer(const cv::Mat& mat)
{

    {
        std::lock_guard<std::mutex> pool_lock(EagleLib::pool::ObjectPool<cv::ogl::Buffer>::mtx);
        // Find any unused opengl buffers of the correct size
        for (auto buf : _pool)
        {
            if (buf->rows() == mat.rows && buf->cols() == mat.cols && buf->type() == mat.type())
            {
                buf->copyFrom(mat, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
                return pool::Ptr<cv::ogl::Buffer>(buf, this);
            }
        }
    }
    
    // Need to create a new opengl buffer
    cv::ogl::Buffer* buf = new cv::ogl::Buffer;
    buf->copyFrom(mat, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
    //used_ogl_buffers.push_back(buf);
    return pool::Ptr<cv::ogl::Buffer>(buf, this);
}
REGISTERSINGLETON(ogl_allocator, true);
