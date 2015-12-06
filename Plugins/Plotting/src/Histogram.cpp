#include "PlottingImpl.hpp"
#include "Plotting.h"
#include "QVector"


using namespace EagleLib;

namespace EagleLib
{
    class HistogramPlotter: public QtPlotter, public StaticPlotPolicy
    {
        QVector<QCPBars*> hists;
        QVector<double> bins;
    public:
        HistogramPlotter();
        virtual QWidget* CreatePlot(QWidget* parent);
        
        virtual bool AcceptsParameter(Parameters::Parameter::Ptr param) const;
        virtual std::string PlotName() const;

        virtual QWidget* GetControlWidget(QWidget* parent);

        virtual void AddPlot(QWidget *plot_);

		virtual void UpdatePlots(); // called from ui thread

        virtual void SetInput(Parameters::Parameter::Ptr param_);

		virtual void OnParameterUpdate(cv::cuda::Stream* stream);
		virtual void Serialize(ISimpleSerializer *pSerializer)
		{
			QtPlotter::Serialize(pSerializer);
			SERIALIZE(hists);
			SERIALIZE(bins);
		}
    };
}
HistogramPlotter::HistogramPlotter()
{

}



QWidget* HistogramPlotter::CreatePlot(QWidget* parent)
{
    QCustomPlot* plot = new QCustomPlot(parent);
    plot->setInteractions((QCP::Interaction)255);
	AddPlot(plot);
    return plot;
}


bool HistogramPlotter::AcceptsParameter(Parameters::Parameter::Ptr param) const
{
    return VectorSizePolicy::acceptsSize(getSize(param));
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

void HistogramPlotter::SetInput(Parameters::Parameter::Ptr param_)
{
    Plotter::SetInput(param_);
	OnParameterUpdate(nullptr);
	for (auto hist : hists)
	{
		hist->setData(bins, data);
	}
    for(auto itr : plots)
    {
        QCustomPlot* plot = dynamic_cast<QCustomPlot*>(itr);
        plot->rescaleAxes();
        plot->replot();
    }
}
void HistogramPlotter::UpdatePlots()
{
	{
		rmt_ScopedCPUSample(HistogramPlotter_UpdatePlots_setData);
		for (auto hist : hists)
		{
			hist->setData(bins, data);
		}
	}
	{
		rmt_ScopedCPUSample(HistogramPlotter_UpdatePlots_replot);

		for (auto plot : plots)
		{
			if (plot->isVisible())
			{
				auto qcp = dynamic_cast<QCustomPlot*>(plot);
				//qcp->rescaleAxes();
				qcp->replot();
			}
			//dynamic_cast<QCustomPlot*>(plot)->replot();
		}
	}
}
void HistogramPlotter::OnParameterUpdate(cv::cuda::Stream* stream)
{
	bool shown = false;
	if (stream == nullptr)
	{
		stream = &plottingStream;
		shown = true;
	}
		
	
	for (auto itr : plots)
	{
		if (itr->isVisible())
		{
			shown = true;
			break;
		}
	}
	if (shown)
	{
		if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
		{
			cv::cuda::GpuMat mat_64f;
			auto typedParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param);
			if (typedParam->Data()->depth() != CV_64F)
				typedParam->Data()->convertTo(mat_64f, CV_64F, *stream);
			else
				mat_64f = *typedParam->Data();
			data.resize(mat_64f.cols);
			cv::Mat dest(1, mat_64f.cols, CV_64F, data.data());
			mat_64f.download(dest, *stream);
			

			if (bins.size() != data.size())
			{
				bins.resize(data.size());
				for (int i = 0; i < bins.size(); ++i)
				{
					bins[i] = i;
				}
			}
			HistogramPlotter* This = this;
			EagleLib::cuda::enqueue_callback_async(
				[This, this]()->void
			{
				Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&HistogramPlotter::UpdatePlots, This));
			}, *stream);
			
		}
	}
}
REGISTERCLASS(HistogramPlotter)

