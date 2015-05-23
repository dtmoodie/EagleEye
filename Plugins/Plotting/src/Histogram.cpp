#include "plotters/Plotter.h"
#include "qcustomplot.h"
#include "Histogram.h"
#include "PlottingImpl.hpp"



using namespace EagleLib;

HistogramPlotter::HistogramPlotter()
{

}
bool HistogramPlotter::acceptsType(EagleLib::Parameter::Ptr param) const
{
    return true;
}
std::string HistogramPlotter::plotName() const
{
    return "HistogramPlotter";
}
QWidget* HistogramPlotter::getSettingsWidget() const
{
    return nullptr;
}
REGISTERCLASS(HistogramPlotter)

