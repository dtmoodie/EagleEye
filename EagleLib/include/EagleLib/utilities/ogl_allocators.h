#pragma once
#include "EagleLib/Detail/Export.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>

#include "GpuMatAllocators.h"
#include <boost/thread.hpp>
#include <EagleLib/utilities/ObjectPool.hpp>
#include <IObject.h>

#include <list>
#include <map>
namespace EagleLib
{
    class ogl_allocator;
    
    class EAGLE_EXPORTS ref_buffer : public cv::ogl::Buffer
    {
        ogl_allocator* allocator;
        friend class ogl_allocator;
    public:

    };

    // Should use the object pooling code to keep track of ogl buffer usage
    class EAGLE_EXPORTS ogl_allocator : public CombinedAllocator, public pool::ObjectPool<cv::ogl::Buffer>, public IObject
    {
    protected:

        // Should use a hash instead, lists all the locations where a gpumat is requested and then later a opengl buffer is requested from that gpumat
        std::list<std::tuple<std::string, int, int, int, boost::thread::id, bool>> buffer_request_locations;

        // Now handled by pool::ObjectPool
        //std::list<cv::ogl::Buffer> unused_ogl_buffers;

        // I don't think this is needed because opengl buffers will be returned to ObjectPool::_pool automatically
        //std::list<cv::ogl::Buffer> used_ogl_buffers;


        // All of the opengl buffers which are currently being mapped to global memory
        // Keep this list so that when get_ogl_buffer is called, we know which buffer to grab, also keeps the buffers from being returned to the object pool
        std::map<unsigned char*, cv::ogl::Buffer*> mapped_buffers;
        ogl_allocator();
        bool is_default_allocator;
    public:

        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);

        
        
        virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        virtual void free(cv::cuda::GpuMat* mat);

        // Creates a GpuMat from an opengl buffer
        virtual cv::cuda::GpuMat createMat(int rows, int cols, int type, cv::cuda::Stream& stream);

        // Either creates a new opengl buffer which then is filled with data from mat, or if mat was correctly
        // created from an opengl buffer, then mat will be unregistered with cuda and can be used in opengl
        pool::Ptr<cv::ogl::Buffer> get_ogl_buffer(const cv::cuda::GpuMat& mat, cv::cuda::Stream& stream);
        pool::Ptr<cv::ogl::Buffer> get_ogl_buffer(const cv::Mat& mat);
    };
}