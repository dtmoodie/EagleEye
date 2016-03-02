#include "IObjectNotifiable.h"
#include "EagleLib/rcc/ObjectManager.h"

IObjectNotifiable::IObjectNotifiable():
    object_swapped(false)
{
    EagleLib::ObjectManager::Instance().register_notifier(this);
}

IObjectNotifiable::~IObjectNotifiable()
{
    EagleLib::ObjectManager::Instance().remove_notifier(this);
}

void IObjectNotifiable::notify_swap()
{
    object_swapped = true;
}