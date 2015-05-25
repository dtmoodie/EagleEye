#include "plotters/Plotter.h"
#include "QVector"
#include "PlottingImpl.hpp"
#include "qcustomplot.h"

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

    };
}
