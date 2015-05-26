#include "Plotting.h"

IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}

void SetupModule(IRuntimeObjectSystem* objectSystem)
{
    objectSystem->AddIncludeDir(PROJECT_INCLUDE_DIR);
#ifdef QT_INCLUDE_DIR
    objectSystem->AddIncludeDir(QT_INCLUDE_DIR);
#endif
}
