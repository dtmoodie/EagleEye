#include "plotters/Plotter.h"

using namespace EagleLib;
Plotter::Plotter()
{

}
Plotter::~Plotter()
{
	bc.disconnect();
}
void Plotter::Init(bool firstInit)
{
	if (!firstInit)
	{
		if (param)
		{
			bc = param->RegisterNotifier(boost::bind(&Plotter::OnParameterUpdate, this, _1));
		}
	}
}
void Plotter::Serialize(ISimpleSerializer *pSerializer)
{
	SERIALIZE(param);
}

void Plotter::SetInput(Parameters::Parameter::Ptr param_)
{
	bc.disconnect();
	param = param_;

	if (param)
		bc = param->RegisterNotifier(boost::bind(&Plotter::OnParameterUpdate, this, _1));
}


void QtPlotter::Serialize(ISimpleSerializer *pSerializer)
{
    Plotter::Serialize(pSerializer);
    SERIALIZE(plots);
}

void QtPlotter::AddPlot(QWidget *plot_)
{
    plots.push_back(plot_);
}
QtPlotter::PlotterType QtPlotter::Type() const
{
	return QT_Plotter;
}