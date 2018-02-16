#include "PlottingImpl.hpp"
#define PARAMETERS_USE_UI
#define Qt5_FOUND
#define OPENCV_FOUND

using namespace EagleLib;

QtPlotterImpl::QtPlotterImpl()
{
    channels = -1;
    controlWidget = nullptr;
    converter = nullptr;
}
QtPlotterImpl::~QtPlotterImpl()
{
}
QWidget* QtPlotterImpl::GetControlWidget(QWidget* parent)
{
    if (parameters.empty())
        return nullptr;
    if (controlWidget == nullptr)
    {
        controlWidget.reset(new QWidget(parent));
        QGridLayout* layout = new QGridLayout(controlWidget.get());
        int row = 0;
        for (auto param : parameters)
        {
            auto proxy = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(param);
            if (proxy)
            {
                parameterProxies.push_back(proxy);
                auto widget = proxy->GetParameterWidget(parent);
                layout->addWidget(widget, row, 0);
                ++row;
            }
        }
        controlWidget->setLayout(layout);
    }
    return controlWidget.get();
}
void QtPlotterImpl::Serialize(ISimpleSerializer* pSerializer)
{
    QtPlotter::Serialize(pSerializer);
    SERIALIZE(converter);
    SERIALIZE(channels);
    SERIALIZE(size);
    SERIALIZE(parameterProxies);
    SERIALIZE(parameters);
    SERIALIZE(controlWidget);
}
void QtPlotterImpl::OnParameterUpdate(cv::cuda::Stream* stream)
{
    if (param == nullptr)
        return;
    if (stream == nullptr)
        stream = &plottingStream;
    if (converter == nullptr)
    {
        converter = Parameters::Converters::Double::Factory::Create(param, stream);
        converter->Update(param, stream);
        return;
    }
    CV_Assert(converter);
    if (converter->GetTypeInfo() != param->GetTypeInfo())
    {
        converter = Parameters::Converters::Double::Factory::Create(param, stream);
        return;
    }
    converter->Update(param, stream);
}
void HistoryPlotter::PlotInit(bool firstInit)
{
    if (firstInit)
    {
        auto param = new Parameters::TypedParameter<size_t>("History Size", 10);
        connections.push_back(param->RegisterNotifier(std::bind(&HistoryPlotter::on_history_size_change, this)));
        parameters.push_back(param);
    }
    else
    {
        connections.push_back(GetParameter<size_t>("History Size")
                                  ->RegisterNotifier(std::bind(&HistoryPlotter::on_history_size_change, this)));
    }
}

HistoryPlotter::HistoryPlotter() : QtPlotterImpl()
{
}

void HistoryPlotter::on_history_size_change()
{
    size_t capacity = *GetParameter<size_t>("History Size")->Data();
    for (auto& itr : channelData)
    {
        itr.set_capacity(capacity);
    }
}
