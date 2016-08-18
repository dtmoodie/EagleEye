#include "EagleLib/rendering/RenderingEngine.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/SystemTable.hpp"

using namespace EagleLib;


void IRenderObjectFactory::RegisterConstructorStatic(std::shared_ptr<IRenderObjectConstructor> constructor)
{
    auto systemTable = PerModuleInterface::GetInstance()->GetSystemTable();
    if (systemTable)
    {
        auto factoryInstance = systemTable->GetSingleton<IRenderObjectFactory>();
        if (factoryInstance)
        {
            factoryInstance->RegisterConstructor(constructor);
        }
    }
}