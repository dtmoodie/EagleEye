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

        virtual QWidget* getPlot(QWidget* parent)
        {
            QTableWidget* widget = new QTableWidget(parent);
            widget->setRowCount(10);
            widget->setColumnCount(10);
            return widget;
        }

        virtual bool acceptsWidget(QWidget *widget)
        {
            return dynamic_cast<QTableWidget*>(widget) != nullptr;
        }

        virtual bool acceptsType(Parameters::Parameter::Ptr param) const
        {
            //return EagleLib::acceptsType<cv::Mat>(param->typeInfo);
			if (Loki::TypeInfo(typeid(cv::Mat)) == param->GetTypeInfo() || Loki::TypeInfo(typeid(cv::cuda::GpuMat)) == param->GetTypeInfo())
			{
				auto size = getSize(param);
				if (size.width < 10 && size.height < 10)
					return true;
			}
			return false;
        }
        virtual std::string plotName() const
        {
            return "MatrixView";
        }
        virtual QWidget* getSettingsWidget() const
        {
            return nullptr;
        }
        virtual void addPlot(QWidget *plot_)
        {
            QTableWidget* widget = dynamic_cast<QTableWidget*>(plot_);
            if(widget)
            {
                plots.push_back(widget);
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
            std::vector<QTableWidgetItem*> items;
            items.reserve(mat->size().area());
			const int channels = mat->channels();
            for(int i = 0; i < mat->rows; ++i)
            {
                for(int j = 0; j < mat->cols; ++j)
                {
                    switch(mat->depth())
                    {
                    case CV_8U:
						items.push_back(readItems(mat->ptr<uchar>(i, j), channels));
                        break;
					case CV_16U:
						items.push_back(readItems(mat->ptr<ushort>(i, j), channels));
						break;
					case CV_16S:
						items.push_back(readItems(mat->ptr<short>(i, j), channels));
						break;
                    case CV_32F:
						items.push_back(readItems(mat->ptr<float>(i, j), channels));
                        break;
                    case CV_64F:
						items.push_back(readItems(mat->ptr<double>(i, j), channels));
                        break;
                    case CV_32S:
						items.push_back(readItems(mat->ptr<int>(i, j), channels));
                        break;
					default: break;
                    }
                }
            }
            for(QTableWidget* widget: plots)
            {
                int count = 0;
				widget->clearContents();
                for(int i = 0; i < mat->rows; ++i)
                {
                    for(int j = 0; j < mat->cols && count < items.size(); ++j, ++count)
                    {
                        widget->setItem(i,j,items[count]);
                    }
                }
            }
        }

        virtual void setInput(Parameters::Parameter::Ptr param_)
        {
            Plotter::setInput(param_);
            try
            {
                doUpdate();
            }catch(cv::Exception &e)
            {
                std::cout << e.what() << std::endl;
            }

        }

    };

}
using namespace EagleLib;
REGISTERCLASS(MatrixView)
