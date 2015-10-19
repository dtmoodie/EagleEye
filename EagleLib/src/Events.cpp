#include "Events.h"

#include "ObjectInterfacePerModule.h"
#include "SystemTable.hpp"

using namespace EagleLib;
SignalHandler::SignalHandler()
{
    auto systemTable = PerModuleInterface::GetInstance()->GetSystemTable();
    if (systemTable)
    {
        systemTable->SetSingleton<ISignalHandler>(this);
    }
}

ISignalManager* SignalHandler::GetSignalManager(Loki::TypeInfo& type)
{
    auto itr = signalManagers.find(type);
    if (itr == signalManagers.end())
        return nullptr;
    return itr->second;
}
ISignalManager* SignalHandler::AddSignalManager(ISignalManager* manager)
{
    signalManagers[manager->GetType()] = manager;
    return manager;
}

REGISTERSINGLETON(SignalHandler, true);