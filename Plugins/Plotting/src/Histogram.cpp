#include "plotters/Plotter.h"
#include "Histogram.h"




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
void HistogramPlotter::doUpdate()
{
    boost::recursive_mutex::scoped_lock lock(param->mtx);
    StaticPlotPolicy::addPlotData(param);
    lock.unlock();
    if(bins.size() != getPlotData().size())
    {
        bins.resize(StaticPlotPolicy::getPlotData().size());
        for(int i = 0; i < bins.size(); ++i)
        {
            bins[i] = i;
        }
    }
    for(int i = 0; i < hists.size();++i)
    {
        hists[i]->setData(bins, StaticPlotPolicy::getPlotData());
    }

    for(int i = 0; i < plots.size(); ++i)
    {
        plots[i]->replot();
    }
}

void HistogramPlotter::addPlot(QCustomPlot *plot_)
{
    QtPlotter::addPlot(plot_);
    QCPBars* hist = new QCPBars(plot_->xAxis, plot_->yAxis);
    hist->setWidth(1);
    hists.push_back(hist);
}

REGISTERCLASS(HistogramPlotter)

