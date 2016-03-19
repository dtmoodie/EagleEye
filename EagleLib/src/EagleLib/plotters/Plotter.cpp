#include "Plotter.h"

using namespace EagleLib;
Plotter::Plotter()
{

}
Plotter::~Plotter()
{
	//bc.disconnect();
}

void Plotter::Init(bool firstInit)
{
	if (!firstInit)
	{
		if (param)
		{
			bc.reset();
			//bcdisconnect();
            bc = param->RegisterNotifier(std::bind(&Plotter::OnParameterUpdate, this, std::placeholders::_1));
		}
	}
}

void Plotter::Serialize(ISimpleSerializer *pSerializer)
{
	IObject::Serialize(pSerializer);
	SERIALIZE(param);
}

void Plotter::SetInput(Parameters::Parameter::Ptr param_)
{
	//bc.disconnect();
	bc.reset();
	param = param_;

	if (param)
        bc = param->RegisterNotifier(std::bind(&Plotter::OnParameterUpdate, this, std::placeholders::_1));
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
