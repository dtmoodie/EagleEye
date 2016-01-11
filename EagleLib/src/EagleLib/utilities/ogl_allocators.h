#pragma once
#include "EagleLib/Defs.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>

#include "GpuMatAllocators.h"
#include <boost/thread.hpp>

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
	class EAGLE_EXPORTS ogl_allocator : public CombinedAllocator
	{
    protected:

        std::list<std::tuple<std::string, int, int, int, boost::thread::id, bool>> buffer_request_locations;
		std::list<cv::ogl::Buffer> unused_ogl_buffers;
        std::list<cv::ogl::Buffer> used_ogl_buffers;
		// All of the opengl buffers which are currently being mapped to global memory
		std::map<unsigned char*, cv::ogl::Buffer> mapped_buffers;
        ogl_allocator(size_t initial_pool_size = 10000000, size_t threshold_level = 1000000);
	public:
        static ogl_allocator* instance(size_t initial_pool_size = 10000000, size_t threshold_level = 1000000);
        
		virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
		virtual void free(cv::cuda::GpuMat* mat);

		// Creates a GpuMat from an opengl buffer
		virtual cv::cuda::GpuMat createMat(int rows, int cols, int type, cv::cuda::Stream& stream);
		cv::ogl::Buffer mapBuffer(cv::cuda::GpuMat& mat, cv::cuda::Stream& stream);
	};
}