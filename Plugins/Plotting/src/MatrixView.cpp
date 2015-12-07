#include "Plotting.h"
#include "PlottingImpl.hpp"


template<typename T> QTableWidgetItem* readItems(T* data, const int channels)
{
	QString str;
	for (int i = 0; i < channels; ++i)
	{
		if (i != 0)
			str += ",";
		str += QString::number(data[i]);
	}
	return new QTableWidgetItem(str);
}

namespace EagleLib
{
    class MatrixView: public QtPlotterImpl, public StaticPlotPolicy
    {
        QVector<QTableWidget*> plots;
    public:
        MatrixView():QtPlotterImpl()
        {

        }
		void Serialize(ISimpleSerializer *pSerializer)
		{
			QtPlotterImpl::Serialize(pSerializer);
			SERIALIZE(plots);
		}
        virtual QWidget* CreatePlot(QWidget* parent)
        {
            QTableWidget* widget = new QTableWidget(parent);
            widget->setRowCount(10);
            widget->setColumnCount(10);
            return widget;
        }

        virtual bool AcceptsParameter(Parameters::Parameter::Ptr param) const
        {
			if (Loki::TypeInfo(typeid(cv::Mat)) == param->GetTypeInfo() || Loki::TypeInfo(typeid(cv::cuda::GpuMat)) == param->GetTypeInfo())
			{
				auto size = getSize(param);
				if (size.width < 10 && size.height < 10)
					return true;
			}
			return false;
        }
        virtual std::string PlotName() const
        {
            return "MatrixView";
        }
        virtual QWidget* GetControlWidget(QWidget* parent)
        {
            return nullptr;
        }
        virtual void AddPlot(QWidget *plot_)
        {
            QTableWidget* widget = dynamic_cast<QTableWidget*>(plot_);
            if(widget)
            {
                plots.push_back(widget);
            }
        }
		virtual void UpdateUi(std::vector<QTableWidgetItem*> items, cv::Size size)
		{
			for (QTableWidget* widget : plots)
			{
				int count = 0;
				widget->clearContents();
				for (int i = 0; i < size.height; ++i)
				{
					for (int j = 0; j < size.width && count < items.size(); ++j, ++count)
					{
						widget->setItem(i, j, items[count]);
					}
				}
			}
		}
        virtual void doUpdate()
        {
            cv::cuda::GpuMat* d_mat = getParameterPtr<cv::cuda::GpuMat>(param);
            cv::Mat h_mat;
            cv::Mat* mat;
            if(d_mat)
            {
                if(d_mat->empty())
                    return;
                d_mat->download(h_mat);
                mat = &h_mat;
            }else
                mat = getParameterPtr<cv::Mat>(param);
            if(mat == nullptr)
                return;
        }

        virtual void SetInput(Parameters::Parameter::Ptr param_)
        {
			QtPlotterImpl::SetInput(param_);
			OnParameterUpdate(nullptr);
        }

		virtual void OnParameterUpdate(cv::cuda::Stream* stream)
		{
			if (param == nullptr)
				return;
			bool shown = false;
			for (auto plot : plots)
			{
				if (plot->isVisible())
				{
					shown = true;
					break;
				}
			}
			
			if (stream == nullptr)
			{
				stream = &plottingStream;
				shown = true;
			}
			if (!shown)
				return;
			QtPlotterImpl::OnParameterUpdate(stream);
			auto This = this;
			auto _converter = converter;
			EagleLib::cuda::enqueue_callback_async(
				[_converter, This]()->void
			{
				std::vector<QTableWidgetItem*> items;
				items.reserve(_converter->NumRows() * _converter->NumCols());
				const int channels = _converter->NumChan();
				for (int i = 0; i < _converter->NumRows(); ++i)
				{
					for (int j = 0; j < _converter->NumCols(); ++j)
					{
						QString text;
						for (int c = 0; c < channels; ++c)
						{
							if (c != 0)
								text += ",";
							text += QString::number(_converter->GetValue(i, j, c));
						}
					}
				}
				Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&MatrixView::UpdateUi, This, items, cv::Size(_converter->NumCols(), _converter->NumRows())));
			}, *stream);
		}
    };

}
using namespace EagleLib;
REGISTERCLASS(MatrixView)
