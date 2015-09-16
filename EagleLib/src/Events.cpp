#include "Events.h"

#include "ObjectInterfacePerModule.h"
#include "SystemTable.hpp"

using namespace EagleLib;

EventHandler::EventHandler()
{
	PerModuleInterface::GetInstance()->GetSystemTable()->eventHandler = this;
}
EventHandler* EventHandler::instance()
{
	return PerModuleInterface::GetInstance()->GetSystemTable()->eventHandler;
}

REGISTERSINGLETON(EventHandler, true);