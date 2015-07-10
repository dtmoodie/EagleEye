//#include "parameterplotter.h"
//#include <boost/circular_buffer.hpp>
//#include "Manager.h"


//QList<ParameterPlotterFactory*> ParameterPlotter::getPlotters(Parameters::Parameter::Ptr param)
//{
//    QList<ParameterPlotterFactory*> output;
//    for(auto it = g_factories.begin(); it != g_factories.end(); ++it)
//    {
//        if((*it)->acceptsType(param))
//            output.push_back(*it);
//    }
//    return output;
//}


//ParameterPlotter* ParameterPlotter::getPlot(Parameters::Parameter::Ptr param)
//{

//}

//ParameterPlotter::ParameterPlotter(Parameters::Parameter::Ptr param_, QCustomPlot *plot_):
//    param(param_), plot(plot_)
//{
//    param->onUpdate.connect(boost::bind(&ParameterPlotter::onUpdate, this));
//}

//ParameterPlotter::~ParameterPlotter()
//{

//}

//void ParameterPlotter::setPlot(QCustomPlot *plot_)
//{
//    if(plot)
//    {

//    }
//    plot = plot_;
//}




//static PlotterFactory<HistogramPlotter, SingleChannelPolicy, VectorSizePolicy, TypePolicy<cv::Mat>, TypePolicy<cv::cuda::GpuMat>, TypePolicy<cv::cuda::HostMem>> histPlotter("HistogramPlotter");
//bool HistogramPlotter::acceptsType(Parameters::Parameter::Ptr param)
//{
//    return EagleLib::acceptsType<cv::Mat>(param->typeInfo) ||
//            EagleLib::acceptsType<cv::cuda::GpuMat>(param->typeInfo) ||
//            EagleLib::acceptsType<cv::cuda::HostMem>(param->typeInfo);
//}

//HistogramPlotter::HistogramPlotter(Parameters::Parameter::Ptr param, QCustomPlot* plot_):
//    ParameterPlotter(param, plot_)
//{
//    connect(this, SIGNAL(update()),this, SLOT(doUpdate()), Qt::QueuedConnection);

//}

//HistogramPlotter::~HistogramPlotter()
//{

//}

//void HistogramPlotter::setPlot(QCustomPlot* plot_)
//{
//    if(plot && graph)
//    {
//        plot->removePlottable(graph);
//    }
//    plot = plot_;
//    if(graph == nullptr)
//    {
//        graph = new QCPBars(plot->xAxis, plot->yAxis);
//    }else
//    {
//        graph->setKeyAxis(plot->xAxis);
//        graph->setValueAxis(plot->yAxis);
//    }
//    plot->addPlottable(graph);
//    graph->setWidth(10);
//    graph->setBrush(QColor(10,140,70,160));
//    graph->setPen(Qt::NoPen);
//    doUpdate();
//}

//void HistogramPlotter::setWindowSize(int size)
//{

//}

//QWidget* HistogramPlotter::getSettingsWidget(QWidget* parent)
//{
//    QWidget* widget = new QWidget(parent);
//    QGridLayout* layout = new QGridLayout(widget);
//    widget->setLayout(layout);
//    QLabel* lbl = new QLabel("Histogram bins: ", widget);
//    layout->addWidget(lbl, 0,0);
//    QComboBox* comboBox = new QComboBox(widget);
//    boost::function<bool(Loki::TypeInfo&)> f(
//                [](Loki::TypeInfo& type)->bool{
//                    return EagleLib::acceptsType<cv::Mat>(type) ||
//                            EagleLib::acceptsType<cv::cuda::GpuMat>(type) ||
//                            EagleLib::acceptsType<cv::cuda::HostMem>(type);});
//    std::vector<std::string> candidates = EagleLib::NodeManager::getInstance().getParametersOfType(f);
//    comboBox->addItem("");
//    for(int i = 0; i < candidates.size(); ++i)
//    {
//        comboBox->addItem(QString::fromStdString(candidates[i]));
//    }
//    connect(comboBox, SIGNAL(currentIndexChanged(QString)), this, SLOT(onScaleChange(QString)));
//    // TODO: Add color other widgets for controlling the look of the histogram plot
//    return widget;
//}
//void HistogramPlotter::onScaleChange(QString name)
//{
//    Parameters::Parameter::Ptr param = EagleLib::NodeManager::getInstance().getParameter(name.toStdString());
//    if(param)
//        scale = getParamArrayData(param,0);
//}

//void HistogramPlotter::doUpdate()
//{
//    StaticPlotPolicy::addPlotData(param);
//    QVector<double> data = StaticPlotPolicy::getPlotData();
//    if(scale.size() != data.size())
//    {
//        scale.resize(data.size());
//        for(int i = 0; i < data.size(); ++i)
//        {
//            scale[i] = i;
//        }
//    }
//    graph->setData(data,scale);
//}
