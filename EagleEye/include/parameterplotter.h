//#ifndef PARAMETERPLOTTER_H
//#define PARAMETERPLOTTER_H
//#include <qcustomplot.h>
//#include "Parameters.h"
//#include "ParameterPlotterImpl.hpp"




//class ParameterPlotter:public QObject
//{
//    Q_OBJECT
//protected:
//    Parameters::Parameter::Ptr param;
//    QCustomPlot* plot;
//public:
//    static QList<ParameterPlotterFactory*> getPlotters(Parameters::Parameter::Ptr param);
//    static bool acceptsType(Parameters::Parameter::Ptr param);
//    static ParameterPlotter* getPlot(Parameters::Parameter::Ptr param);

//    ParameterPlotter(Parameters::Parameter::Ptr param, QCustomPlot* plot_ = nullptr);
//    ~ParameterPlotter();

//    virtual void setPlot(QCustomPlot* plot_);
//    virtual void setWindowSize(int size) = 0;

//    virtual QWidget* getSettingsWidget(QWidget* parent) = 0;
//    virtual void onUpdate() = 0;
//};



//// proxy interface class to allow templating parts of the plotting code
//class IParameterPlotterProxy
//{

//};

//class HistogramPlotter: public ParameterPlotter, public StaticPlotPolicy
//{
//    QCPBars* graph;
//    Q_OBJECT
//    QVector<double> scale;
//public:
//    static bool acceptsType(Parameters::Parameter::Ptr param);
//    HistogramPlotter(Parameters::Parameter::Ptr param, QCustomPlot* plot_);
//    ~HistogramPlotter();

//    virtual void setPlot(QCustomPlot* plot_);
//    virtual void setWindowSize(int size);

//    virtual QWidget* getSettingsWidget(QWidget* parent);
//    virtual void onUpdate()
//    {        emit update();    }

//signals:
//    void update();

//private slots:
//    virtual void doUpdate();
//    virtual void onScaleChange(QString name);
//};

//#endif // PARAMETERPLOTTER_H
