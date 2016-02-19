#include "WindowCallbackManager.h"
#include "UiCallbackHandlers.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/ObjectManager.h"
#include "EagleLib/rcc/SystemTable.hpp"
using namespace EagleLib;

WindowCallbackHandlerManager::WindowCallbackHandlerManager()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        table->SetSingleton<WindowCallbackHandlerManager>(this);
    }
}
void WindowCallbackHandlerManager::Serialize(ISimpleSerializer* pSerializer)
{
    SERIALIZE(instances);
}
void WindowCallbackHandlerManager::Init(bool firstInit)
{
    if (firstInit)
    {

    }
    else
    {
        for (auto& itr : instances)
        {
            itr.second->set_stream(itr.first);
        }
    }
}
WindowCallbackHandler* WindowCallbackHandlerManager::instance(size_t stream_id)
{
	std::lock_guard<std::mutex> lock(mtx);
    auto itr = instances.find(stream_id);
    if (itr == instances.end())
    {
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto manager = table->GetSingleton<ObjectManager>();
        auto ptr = manager->GetObject<WindowCallbackHandler, IID_IOBJECT>("WindowCallbackHandler");
        ptr->set_stream(stream_id);
        instances[stream_id] = ptr;
        return ptr.get();
    }
    return itr->second.get();
}
REGISTERSINGLETON(WindowCallbackHandlerManager, true)