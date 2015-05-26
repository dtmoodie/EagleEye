#include "Manager.h"

#ifdef __cplusplus
extern "C"{
#endif
    IPerModuleInterface* GetModule();
    void SetupModule(IRuntimeObjectSystem* objectSystem);

#ifdef __cplusplus
}
#endif
