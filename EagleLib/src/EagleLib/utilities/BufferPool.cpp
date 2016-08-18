#include "EagleLib/utilities/BufferPool.hpp"
#include <time.h>
#include <iostream>

#include "MetaObject/Thread/InterThread.hpp"

using namespace EagleLib;

void EagleLib::scoped_buffer_dallocator_callback(int status, void* user_data)
{
    auto mat_ptr = static_cast<cv::cuda::GpuMat*>(user_data);
    scoped_buffer::deallocateQueue.push(mat_ptr);
}
scoped_buffer::scoped_buffer(cv::cuda::Stream stream)
{
    data = new cv::cuda::GpuMat();
    this->stream = stream;
}
scoped_buffer::~scoped_buffer()
{
    stream.enqueueHostCallback(EagleLib::scoped_buffer_dallocator_callback, data);
}
cv::cuda::GpuMat& scoped_buffer::GetMat()
{
    return *data;
}

boost::lockfree::queue<cv::cuda::GpuMat*> scoped_buffer::deallocateQueue(500);

scoped_buffer::GarbageCollector::GarbageCollector()
{
    
}
void scoped_buffer::GarbageCollector::Run()
{
    cv::cuda::GpuMat* mat;
    while (scoped_buffer::deallocateQueue.pop(mat))
    {
        delete mat;
    }
}