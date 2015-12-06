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
    class MatrixView: public QtPlotter, public StaticPlotPolicy
    {
        QVector<QTableWidget*> plots;
    public:
        MatrixView()
        {

        }
		void Serialize(ISimpleSerializer *pSerializer)
		{
			QtPlotter::Serialize(pSerializer);
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
            Plotter::SetInput(param_);
			OnParameterUpdate(nullptr);
        }
		virtual void OnParameterUpdate(cv::cuda::Stream* stream)
		{
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
			cv::Mat mat_data;
			if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
			{
				auto typedParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param);
				auto mat = typedParam->Data();
				if (mat->empty())
					return;
				mat->download(mat_data, *stream);
			}
			if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::Mat)))
			{
				auto typedParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::Mat>>(param);
				mat_data = *typedParam->Data();
			}
			if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::HostMem)))
			{
				auto typedParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::HostMem>>(param);
				mat_data = typedParam->Data()->createMatHeader();
			}
			auto This = this;
			EagleLib::cuda::enqueue_callback_async(
				[mat_data, This]()->void
			{
				std::vector<QTableWidgetItem*> items;
				items.reserve(mat_data.size().area());
				const int channels = mat_data.channels();
				for (int i = 0; i < mat_data.rows; ++i)
				{
					for (int j = 0; j < mat_data.cols; ++j)
					{
						switch (mat_data.depth())
						{
						case CV_8U:
							items.push_back(readItems(mat_data.ptr<uchar>(i, j), channels));
							break;
						case CV_16U:
							items.push_back(readItems(mat_data.ptr<ushort>(i, j), channels));
							break;
						case CV_16S:
							items.push_back(readItems(mat_data.ptr<short>(i, j), channels));
							break;
						case CV_32F:
							items.push_back(readItems(mat_data.ptr<float>(i, j), channels));
							break;
						case CV_64F:
							items.push_back(readItems(mat_data.ptr<double>(i, j), channels));
							break;
						case CV_32S:
							items.push_back(readItems(mat_data.ptr<int>(i, j), channels));
							break;
						default: break;
						}
					}
				}
				Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&MatrixView::UpdateUi, This, items, mat_data.size()));
			}, *stream);
		}
    };

}
using namespace EagleLib;
REGISTERCLASS(MatrixView)
