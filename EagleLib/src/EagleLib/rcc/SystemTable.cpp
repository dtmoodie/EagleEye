#include "SystemTable.hpp"

SystemTable::SystemTable()
{

}
void SystemTable::DeleteSingleton(Loki::TypeInfo type)
{
    g_singletons.erase(type);
}