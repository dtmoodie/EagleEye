#include "Signals.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/SystemTable.hpp"

using namespace EagleLib;
SignalManager* SignalManager::get_instance()
{
	auto system_table = PerModuleInterface::GetInstance()->GetSystemTable();
	static SignalManager* g_instance = nullptr;
	if (g_instance == nullptr)
	{
		if (system_table)
		{
			g_instance = system_table->GetSingleton<SignalManager>();
		}
		if (g_instance == nullptr)
		{
			g_instance = new SignalManager();
			system_table->SetSingleton<SignalManager>(g_instance);
		}
	}
	return g_instance;	
}