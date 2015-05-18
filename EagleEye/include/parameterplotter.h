#ifndef PARAMETERPLOTTER_H
#define PARAMETERPLOTTER_H
#include <qcustomplot.h>
#include "Parameters.h"






class ParameterPlotter:public QObject
{
    Q_OBJECT

    QCustomPlot* plot;
public:
    static bool isPlottable(EagleLib::Parameter::Ptr param);
    static ParameterPlotter* getPlot(EagleLib::Parameter::Ptr param);

    ParameterPlotter(EagleLib::Parameter::Ptr param, QCustomPlot* plot_ = nullptr);
    ~ParameterPlotter();

    virtual void setPlot(QCustomPlot* plot_);
    virtual void setWindowSize(int size) = 0;

    virtual QWidget* getSettingsWidget() = 0;
};

#endif // PARAMETERPLOTTER_H
