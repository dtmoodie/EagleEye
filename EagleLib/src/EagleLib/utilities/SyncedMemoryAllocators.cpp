#include "EagleLib/utilities/SyncedMemoryAllocators.hpp"
#include <boost/thread/tss.hpp>

using namespace EagleLib;


static boost::thread_specific_ptr<ISyncedMemoryAllocator> thread_instance;

ISyncedMemoryAllocator* ISyncedMemoryAllocator::Instance()
{
    if(thread_instance.get() == nullptr)
    {
        //thread_instance.reset(new ISyncedMemoryAllocator());
    }
    return thread_instance.get();
}

void ISyncedMemoryAllocator::SetInstance(ISyncedMemoryAllocator* allocator)
{
    thread_instance.reset(allocator);
}
