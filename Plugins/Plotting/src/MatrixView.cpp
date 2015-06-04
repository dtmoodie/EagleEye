#include "Plotting.h"
#include "PlottingImpl.hpp"

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

        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const
        {
            return EagleLib::acceptsType<cv::Mat>(param->typeInfo);

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
            cv::Mat* mat = EagleLib::getParameterPtr<cv::Mat>(param);
            if(mat == nullptr)
                return;
            std::vector<QTableWidgetItem*> items;
            items.reserve(mat->size().area());
            for(int i = 0; i < mat->rows; ++i)
            {
                for(int j = 0; j < mat->cols; ++j)
                {
                    switch(mat->type())
                    {
                    case CV_8U:
                        items.push_back(new QTableWidgetItem(QString::number(mat->at<uchar>(i,j))));
                        break;
                    case CV_32F:
                        items.push_back(new QTableWidgetItem(QString::number(mat->at<float>(i,j))));
                        break;
                    case CV_64F:
                        items.push_back(new QTableWidgetItem(QString::number(mat->at<double>(i,j))));
                        break;
                    case CV_32S:
                        items.push_back(new QTableWidgetItem(QString::number(mat->at<int>(i,j))));
                        break;
                    }
                }
            }
            for(QTableWidget* widget: plots)
            {
                int count = 0;
                for(int i = 0; i < mat->rows; ++i)
                {
                    for(int j = 0; j < mat->cols; ++j, ++count)
                    {
                        widget->setItem(i,j,items[count]);
                    }
                }
            }
        }

        virtual void setInput(Parameter::Ptr param_)
        {
            Plotter::setInput(param_);
            doUpdate();
        }

    };

}
using namespace EagleLib;
REGISTERCLASS(MatrixView)
