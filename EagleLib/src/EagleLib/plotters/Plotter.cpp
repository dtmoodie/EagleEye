#include "Plotter.h"

using namespace EagleLib;

int PlotterInfo::GetObjectInfoType()
{
    return plotter;
}
Plotter::Plotter()
{
    param = nullptr;
}
Plotter::~Plotter()
{
}

void Plotter::Init(bool firstInit)
{
    PlotInit(firstInit);
    ParameteredIObject::Init(firstInit);
    if (!firstInit)
    {
        SetInput(param);
        if (param)
        {
            param->RegisterNotifier(std::bind(&Plotter::OnParameterUpdate, this, std::placeholders::_1));
        }
    }
}
void Plotter::PlotInit(bool firstInit)
{

}

void Plotter::Serialize(ISimpleSerializer *pSerializer)
{
    IObject::Serialize(pSerializer);
    SERIALIZE(param);
}

void Plotter::SetInput(Parameters::Parameter* param_)
{
    param = param_;
    if (param)
    {
        _connections[&param->update_signal] = param->RegisterNotifier(std::bind(&Plotter::OnParameterUpdate, this, std::placeholders::_1));
        _connections[&param->delete_signal] = param->RegisterDeleteNotifier(std::bind(&Plotter::on_param_delete, this, std::placeholders::_1));
    }
}
void Plotter::on_param_delete(Parameters::Parameter* param)
{
    param = nullptr;
}
void QtPlotter::Serialize(ISimpleSerializer *pSerializer)
{
    Plotter::Serialize(pSerializer);
    SERIALIZE(plot_widgets);
    SERIALIZE(_pimpl);
}

void QtPlotter::AddPlot(QWidget* plot_)
{
    plot_widgets.push_back(plot_);
}

QtPlotter::PlotterType QtPlotter::Type() const
{
    return QT_Plotter;
}
#ifdef QT_GUI_LIB
#define Qt5_FOUND
#define HAVE_OPENCV
#include <parameters/UI/Qt.hpp>
#include <parameters/UI/Qt/IParameterProxy.hpp>
class QtPlotter::impl
{
public:
    std::map<std::string, std::shared_ptr<Parameters::UI::qt::IParameterProxy>> parameter_proxies;
    QGridLayout* control_layout;
    QWidget* parent;
};
QWidget* QtPlotter::GetControlWidget(QWidget* parent)
{
    auto parameters = getDisplayParameters();
    if(parameters.size())
    {
        if(_pimpl == nullptr)
        {
            _pimpl.reset(new impl());
        }
        QWidget* control_widget = new QWidget(parent);
        QGridLayout* layout = new QGridLayout();
        control_widget->setLayout(layout);
        for(auto param : parameters)
        {
            auto proxy = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(param);
            auto param_widget = proxy->GetParameterWidget(parent);
            layout->addWidget(param_widget);
            _pimpl->parameter_proxies[param->GetTreeName()] = proxy;
        }
        _pimpl->control_layout = layout;
        _pimpl->parent = parent;
        return control_widget;
    }else
    {
        return nullptr;
    }
}

Parameters::Parameter* QtPlotter::addParameter(Parameters::Parameter* param)
{
    auto ret = Plotter::addParameter(param);
    if(_pimpl)
    {
        auto itr = _pimpl->parameter_proxies.find(param->GetTreeName());
        if(itr == _pimpl->parameter_proxies.end())
        {
            auto proxy = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(param);
            auto param_widget = proxy->GetParameterWidget(_pimpl->parent);
            _pimpl->control_layout->addWidget(param_widget);
            _pimpl->parameter_proxies[param->GetTreeName()] = proxy;
        }else
        {
            itr->second->SetParameter(param);
        }
    }
    return ret;
}

Parameters::Parameter* QtPlotter::addParameter(std::shared_ptr<Parameters::Parameter> param)
{
    auto ret = Plotter::addParameter(param);
    if(_pimpl)
    {
        auto itr = _pimpl->parameter_proxies.find(param->GetTreeName());
        if(itr == _pimpl->parameter_proxies.end())
        {
            auto proxy = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(param.get());
            auto param_widget = proxy->GetParameterWidget(_pimpl->parent);
            _pimpl->control_layout->addWidget(param_widget);
            _pimpl->parameter_proxies[param->GetTreeName()] = proxy;
        }else
        {
            itr->second->SetParameter(param.get());
        }
    }
    return ret;
}
#else
QWidget* QtPlotter::GetControlWidget(QWidget* parent)
{
    return nullptr;
}
Parameters::Parameter* QtPlotter::addParameter(std::shared_ptr<Parameters::Parameter> param)
{
    return Plotter::addParameter(param);
}
Parameters::Parameter* QtPlotter::addParameter(Parameters::Parameter* param)
{
    return Plotter::addParameter(param);
}
#endif