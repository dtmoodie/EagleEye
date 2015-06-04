#include "Plotting.h"

using namespace EagleLib;
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

//QWidget* QCustomPlotter::getPlot()
//{
//    QCustomPlot* plot = new QCustomPlot(this);
//    QCPPlotTitle* title = new QCPPlotTitle(plot, QString::fromStdString(this->plotName()));
    
//    plot->setInteractions((QCP::Interaction)255);
//    plot->plotLayout()->insertRow(0);
//    plot->plotLayout()->addElement(0,0,title);
//    return plot;
//}
