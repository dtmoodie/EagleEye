#include "BufferPool.hpp"
#include <time.h>
#include <iostream>

#include "UI/InterThread.hpp"

using namespace EagleLib;

void EagleLib::scoped_buffer_dallocator_callback(int status, void* user_data)
{
	auto mat_ptr = static_cast<cv::cuda::GpuMat*>(user_data);
	scoped_buffer::deallocateQueue.push(mat_ptr);

	//delete mat_ptr;
	std::cout << "1 " << clock() << std::endl;
}
scoped_buffer::scoped_buffer(cv::cuda::Stream stream)
{
	data = new cv::cuda::GpuMat();
	this->stream = stream;
}
scoped_buffer::~scoped_buffer()
{
	stream.enqueueHostCallback(EagleLib::scoped_buffer_dallocator_callback, data);
	/*cv::cuda::Event ev;
	ev.record(stream);
	allocator->deallocateAfterEvent(data, ev);*/
}
cv::cuda::GpuMat& scoped_buffer::GetMat()
{
	return *data;
}

boost::lockfree::queue<cv::cuda::GpuMat*> scoped_buffer::deallocateQueue(120);

scoped_buffer::GarbageCollector::GarbageCollector()
{
	//Parameters::UI::ProcessingThreadCallbackService::post(boost::bind(&Launcher::Run));
}
void scoped_buffer::GarbageCollector::Run()
{
	cv::cuda::GpuMat* mat;
	while (scoped_buffer::deallocateQueue.pop(mat))
	{
		delete mat;
	}
	
	//Parameters::UI::ProcessingThreadCallbackService::post(boost::bind(&Launcher::Run));
}