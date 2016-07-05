#include "WindowCallbackManager.h"
#include "UiCallbackHandlers.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/ObjectManager.h"
#include "EagleLib/rcc/SystemTable.hpp"
using namespace EagleLib;

/*WindowCallbackHandlerManager::WindowCallbackHandlerManager()
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
            
        }
    }
}
WindowCallbackHandler* WindowCallbackHandlerManager::instance(SignalManager* mgr)
{
    
}
REGISTERSINGLETON(WindowCallbackHandlerManager, true)
*/