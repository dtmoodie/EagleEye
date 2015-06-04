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
        virtual QWidget* getPlot(QWidget* parent);
        virtual bool acceptsWidget(QWidget *widget);
        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const;
        virtual std::string plotName() const;
        virtual QWidget* getSettingsWidget() const;
        virtual void addPlot(QWidget *plot_);
        virtual void doUpdate();
        virtual void setInput(Parameter::Ptr param_);

    };
}
HistogramPlotter::HistogramPlotter()
{

}

QWidget* HistogramPlotter::getPlot(QWidget* parent)
{
    QCustomPlot* plot = new QCustomPlot(parent);
    plot->setInteractions((QCP::Interaction)255);
    return plot;
}

bool HistogramPlotter::acceptsWidget(QWidget *widget)
{
    return dynamic_cast<QCustomPlot*>(widget) != nullptr;
}

bool HistogramPlotter::acceptsType(EagleLib::Parameter::Ptr param) const
{
    return VectorSizePolicy::acceptsSize(getSize(param));
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
    if(param == nullptr)
        return;
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
        dynamic_cast<QCustomPlot*>(plots[i])->replot();
    }
}

void HistogramPlotter::addPlot(QWidget *plot_)
{
    QCustomPlot* _plot = dynamic_cast<QCustomPlot*>(plot_);
    if(_plot == nullptr)
        return;
    QtPlotter::addPlot(plot_);
    QCPBars* hist = new QCPBars(_plot->xAxis, _plot->yAxis);
    hist->setWidth(1);
    _plot->addPlottable(hist);
    _plot->rescaleAxes();
    _plot->replot();
    if(_plot->plotLayout()->rowCount() == 0)
    {
        QCPPlotTitle* title = new QCPPlotTitle(_plot, QString::fromStdString(this->plotName()));
        _plot->plotLayout()->insertRow(0);
        _plot->plotLayout()->addElement(0,0, title);
    }
    hists.push_back(hist);
}
void HistogramPlotter::setInput(Parameter::Ptr param_)
{
    Plotter::setInput(param_);
    doUpdate();
    for(size_t i = 0; i < plots.size(); ++i)
    {
        QCustomPlot* plot = dynamic_cast<QCustomPlot*>(plots[i]);
        plot->rescaleAxes();
        plot->replot();
    }
}
REGISTERCLASS(HistogramPlotter)

