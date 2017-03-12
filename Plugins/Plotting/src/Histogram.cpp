#include "PlottingImpl.hpp"
#include "Plotting.h"
#include "QVector"


using namespace EagleLib;

namespace EagleLib
{
    class HistogramPlotter: public QtPlotterImpl, public StaticPlotPolicy
    {
        QVector<QCPBars*> hists;
        QVector<double> bins;
    public:
        HistogramPlotter();
        virtual QWidget* CreatePlot(QWidget* parent);
        
        virtual bool AcceptsParameter(Parameters::Parameter* param);
        virtual std::string PlotName() const;

        virtual QWidget* GetControlWidget(QWidget* parent);

        virtual void AddPlot(QWidget *plot_);

        virtual void UpdatePlots(bool rescale = false); // called from ui thread

        virtual void SetInput(Parameters::Parameter* param_);

        virtual void OnParameterUpdate(cv::cuda::Stream* stream);
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            QtPlotterImpl::Serialize(pSerializer);
            SERIALIZE(hists);
            SERIALIZE(bins);
        }
        virtual void RescalePlots()
        {

        }
    };
    struct HistogramPlotterInfo: public PlotterInfo
    {
        virtual Plotter::PlotterType GetPlotType()
        {
            return Plotter::QT_Plotter;
        }
        virtual bool AcceptsParameter(Parameters::Parameter* param)
        {
            return VectorSizePolicy::acceptsSize(getSize(param));
        }
        virtual std::string GetObjectName()
        {
            return "HistogramPlotter";
        }
        virtual std::string GetObjectTooltip()
        {
            return "Used for plotting histograms, input must be a calculated histogram";
        }
        virtual std::string GetObjectHelp()
        {
            return "Input must be a parameter that can be converted to a vector, it is then plotted as a histogram";
        }
    };
}

HistogramPlotterInfo hist_info;

HistogramPlotter::HistogramPlotter() :
    QtPlotterImpl()
{

}



QWidget* HistogramPlotter::CreatePlot(QWidget* parent)
{
    QCustomPlot* plot = new QCustomPlot(parent);
    plot->setInteractions((QCP::Interaction)255);
    AddPlot(plot);
    return plot;
}


bool HistogramPlotter::AcceptsParameter(Parameters::Parameter* param)
{
    return hist_info.AcceptsParameter(param);
}
std::string HistogramPlotter::PlotName() const
{
    return "HistogramPlotter";
}
QWidget* HistogramPlotter::GetControlWidget(QWidget* parent)
{
    return nullptr;
}

void HistogramPlotter::AddPlot(QWidget *plot_)
{
    QCustomPlot* _plot = dynamic_cast<QCustomPlot*>(plot_);
    if(_plot == nullptr)
        return;
    QtPlotter::AddPlot(plot_);
    QCPBars* hist = new QCPBars(_plot->xAxis, _plot->yAxis);
    hist->setWidth(1);
    _plot->addPlottable(hist);
    _plot->rescaleAxes();
    _plot->replot();
    if(_plot->plotLayout()->rowCount() == 0)
    {
        QCPPlotTitle* title = new QCPPlotTitle(_plot, QString::fromStdString(this->PlotName()));
        _plot->plotLayout()->insertRow(0);
        _plot->plotLayout()->addElement(0,0, title);
    }
    hists.push_back(hist);
}

void HistogramPlotter::SetInput(Parameters::Parameter* param_)
{
    Plotter::SetInput(param_);
    OnParameterUpdate(nullptr);
}

void HistogramPlotter::UpdatePlots(bool rescale)
{
    {
        rmt_ScopedCPUSample(HistogramPlotter_UpdatePlots_setData);
        data.resize(converter->NumCols());
        for (int i = 0; i < converter->NumCols(); ++i)
        {
            data[i] = converter->GetValue(0, i);
        }
        for (auto hist : hists)
        {
            hist->setData(bins, data);
        }
    }
    {
        rmt_ScopedCPUSample(HistogramPlotter_UpdatePlots_replot);

        for (auto plot : plot_widgets)
        {
            if (plot->isVisible() || rescale)
            {
                auto qcp = dynamic_cast<QCustomPlot*>(plot);
                qcp->replot();
                if (rescale)
                    qcp->rescaleAxes();
                
            }
        }
    }
}
void HistogramPlotter::OnParameterUpdate(cv::cuda::Stream* stream)
{
    bool rescale = stream == nullptr;
    if (rescale)
        stream = &plottingStream;
    if (param == nullptr)
        return;
    bool shown = false;
    for (auto itr : plot_widgets)
    {
        if (itr->isVisible())
        {
            shown = true;
            break;
        }
    }
    if (shown || rescale)
    {
        QtPlotterImpl::OnParameterUpdate(stream);
        if (bins.size() != converter->NumCols())
        {
            bins.resize(converter->NumCols());
            for (int i = 0; i < converter->NumCols(); ++i)
            {
                bins[i] = i;
            }
        }
        EagleLib::cuda::enqueue_callback_async(
            [this, rescale]()->void
        {
            //mo::UI::UiCallbackService::Instance()->post(std::bind(&HistogramPlotter::UpdatePlots, this, rescale));
        }, *stream);
            
    }
}
REGISTERCLASS(HistogramPlotter, &hist_info)

