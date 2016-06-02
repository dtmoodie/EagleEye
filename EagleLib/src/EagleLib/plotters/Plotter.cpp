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
	if (!firstInit)
	{
		SetInput(param);
		if (param)
		{
            param->RegisterNotifier(std::bind(&Plotter::OnParameterUpdate, this, std::placeholders::_1));
		}
	}
	PlotInit(firstInit);
	ParameteredIObject::Init(firstInit);
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
}

void QtPlotter::AddPlot(QWidget* plot_)
{
	plot_widgets.push_back(plot_);
}

QtPlotter::PlotterType QtPlotter::Type() const
{
	return QT_Plotter;
}
