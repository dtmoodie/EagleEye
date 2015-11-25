#include "CudaCallbacks.hpp"


void EagleLib::cuda::ICallback::cb_func(int status, void* user_data)
{
	auto cb = static_cast<ICallback*>(user_data);
	cb->run();
	delete cb;
}
