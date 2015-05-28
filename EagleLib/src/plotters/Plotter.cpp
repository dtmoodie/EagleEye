#include "plotters/Plotter.h"

using namespace EagleLib;



void QtPlotter::Serialize(ISimpleSerializer *pSerializer)
{
    Plotter::Serialize(pSerializer);
    SERIALIZE(plots);
}

void QtPlotter::addPlot(QWidget *plot_)
{
    plots.push_back(plot_);
}
