#include "Manager.h"
#include "plotters/Plotter.h"
#include "qcustomplot.h"

#ifdef __cplusplus
extern "C"{
#endif
    IPerModuleInterface* GetModule();
    void SetupModule(IRuntimeObjectSystem* objectSystem);

#ifdef __cplusplus
}
#endif
//class QCustomPlotter: public EagleLib::QtPlotter
//{
//protected:
    
//public:
//    virtual QWidget* getPlot();
    
//};
