#include "Plotting.h"
#include "PlottingImpl.hpp"
#include "Remotery.h"
#include <parameters/ParameteredObjectImpl.hpp>
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
        //QVector<QTableWidget*> plots;
    public:
        MatrixView():QtPlotterImpl()
        {

        }
		void PlotInit(bool firstInit)
		{
			if (firstInit)
			{
				parameters.push_back(new Parameters::TypedParameter<unsigned int>("Row", 0));
				parameters.push_back(new Parameters::TypedParameter<unsigned int>("Col", 0));
			}
		}
		void Serialize(ISimpleSerializer *pSerializer)
		{
			QtPlotterImpl::Serialize(pSerializer);
		}
        virtual QWidget* CreatePlot(QWidget* parent)
        {
            QTableWidget* widget = new QTableWidget(parent);
            widget->setRowCount(10);
            widget->setColumnCount(10);
            return widget;
        }

        virtual bool AcceptsParameter(Parameters::Parameter::Ptr param)
        {
			auto tmp_converter = Parameters::Converters::Double::Factory::Create(param.get(), &plottingStream);
			if (tmp_converter == nullptr)
				return false;
			delete tmp_converter;
			return true;
        }
        virtual std::string PlotName() const
        {
            return "MatrixView";
        }
        virtual void AddPlot(QWidget *plot_)
        {
            QTableWidget* widget = dynamic_cast<QTableWidget*>(plot_);
            if(widget)
            {
				plot_widgets.push_back(widget);
            }
        }
		virtual void UpdateUi(std::vector<QTableWidgetItem*> items, cv::Size size)
		{
			int row = *GetParameter<unsigned int>("Row")->Data();
			int col = *GetParameter<unsigned int>("Col")->Data();
			for (auto plot : plot_widgets)
			{
				auto widget = dynamic_cast<QTableWidget* > (plot);
				int count = 0;
				widget->clearContents();
				QStringList columnHeader;

				for (int i = row; i < size.height && i < row + 10; ++i)
				{
					
					for (int j = col; j < size.width && j < col + 10 && count < items.size(); ++j, ++count)
					{
						if(i == row)
							columnHeader << QString::number(j);
						widget->setItem(i-row, j-col, items[count]);
					}
				}
				
				widget->setHorizontalHeaderLabels(columnHeader);
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
			rmt_ScopedCPUSample(MatrixView_OnParameterUpdate);
			if (param == nullptr)
				return;
			bool shown = false;
			for (auto plot : plot_widgets)
			{
				auto widget = dynamic_cast<QTableWidget* > (plot);
				if (widget->isVisible())
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
			
			int row = *GetParameter<unsigned int>("Row")->Data();
			int col = *GetParameter<unsigned int>("Col")->Data();
			converter->roi = cv::Rect(col, row, 10, 10);
			EagleLib::cuda::enqueue_callback_async(
				[this, row, col]()->void
			{
				std::vector<QTableWidgetItem*> items;
				items.reserve(converter->NumRows() * converter->NumCols());
				const int channels = converter->NumChan();

				for (int i = row; i < converter->NumRows() && i < row + 10; ++i)
				{
					for (int j = col; j < converter->NumCols() && j < col + 10; ++j)
					{
						QString text;
						for (int c = 0; c < channels; ++c)
						{
							if (c != 0)
								text += ",";
							text += QString::number(converter->GetValue(i, j, c));
						}
						items.push_back(new QTableWidgetItem(text));
					}
				}
                Parameters::UI::UiCallbackService::Instance()->post(std::bind(&MatrixView::UpdateUi, this, items, cv::Size(converter->NumCols(), converter->NumRows())));
			}, *stream);
		}
    };

}
using namespace EagleLib;
REGISTERCLASS(MatrixView)
