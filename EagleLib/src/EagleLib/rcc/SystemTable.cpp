#include "EagleLib/rcc/SystemTable.hpp"

SystemTable::SystemTable()
{

}
void SystemTable::DeleteSingleton(mo::TypeInfo type)
{
    g_singletons.erase(type);
}