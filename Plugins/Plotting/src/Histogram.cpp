#include "plotters/Plotter.h"
#include "QVector"
#include "PlottingImpl.hpp"
#include "qcustomplot.h"

using namespace EagleLib;

namespace EagleLib
{
    class HistogramPlotter: public QtPlotter, public StaticPlotPolicy
    {
        QVector<QCPBars*> hists;
        QVector<double> bins;
    public:
        HistogramPlotter();
        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const;
        virtual std::string plotName() const;
        virtual QWidget* getSettingsWidget() const;
        virtual void addPlot(QCustomPlot *plot_);
        virtual void doUpdate();
        virtual void setInput(Parameter::Ptr param_);

    };
}
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

    for(size_t i = 0; i < plots.size(); ++i)
    {
        plots[i]->replot();
    }
}

void HistogramPlotter::addPlot(QCustomPlot *plot_)
{
    QtPlotter::addPlot(plot_);
    QCPBars* hist = new QCPBars(plot_->xAxis, plot_->yAxis);
    hist->setWidth(1);
    plot_->addPlottable(hist);
    plot_->rescaleAxes();
    plot_->replot();
    hists.push_back(hist);
}
void HistogramPlotter::setInput(Parameter::Ptr param_)
{
    Plotter::setInput(param_);
    doUpdate();
    for(size_t i = 0; i < plots.size(); ++i)
    {
        plots[i]->rescaleAxes();
        plots[i]->replot();
    }
}
REGISTERCLASS(HistogramPlotter)

