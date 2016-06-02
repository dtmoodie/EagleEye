#include "vtkPlotter.h"
#include "vtkLogRedirect.h"
#include <vtkOpenGLRenderWindow.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkAxesActor.h>
#include <QVTKWidget2.h>
#include <qopengl.h>
#include "QtOpenGL/QGLContext"
#include <QOpenGLContext>
using namespace EagleLib;
using namespace EagleLib::Plotting;

Plotter::PlotterType vtkPlotterInfo::GetPlotType()
{
	return Plotter::PlotterType(Plotter::QT_Plotter + Plotter::VTK_Plotter);
}
vtkPlotter::vtkPlotter():
	QtPlotter()
{
	renderer = vtkSmartPointer<vtkRenderer>::New();
	vtkLogRedirect::init();
}

vtkPlotter::~vtkPlotter()
{
	Signals::thread_specific_queue::remove_from_queue(this);
}
bool vtkPlotter::AcceptsParameter(Parameters::Parameter* param)
{
	return false;
}
std::string vtkPlotter::PlotName() const
{
	return "vtkPlotter";
}

void vtkPlotter::SetInput(Parameters::Parameter* param_)
{
	QtPlotter::SetInput(param_);
}

void vtkPlotter::OnParameterUpdate(cv::cuda::Stream* stream)
{

}

void vtkPlotter::AddPlot(QWidget* plot_)
{
	auto widget = dynamic_cast<QVTKWidget2*>(plot_);
	if (widget)
	{
		widget->GetRenderWindow()->AddRenderer(renderer);
	}
}

QWidget* vtkPlotter::CreatePlot(QWidget* parent)
{
	QOpenGLContext* draw_context = new QOpenGLContext();

#if QT_VERSION > 0x050500
	auto global_context = QOpenGLContext::globalShareContext();
	draw_context->setShareContext(global_context);
#endif
	auto widget = new QVTKWidget2(QGLContext::fromOpenGLContext(draw_context), parent);
	widget->makeCurrent();
	widget->show();
	widget->setMinimumWidth(100);
	widget->setMinimumHeight(100);
	render_widgets.push_back(widget);
	widget->GetRenderWindow()->OpenGLInit();
	widget->GetRenderWindow()->AddRenderer(renderer);

	return widget;
}
void vtkPlotter::Init(bool firstInit)
{
	QtPlotter::Init(firstInit);
	if(firstInit)
	{
		vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

		this->renderer->AddActor(axes);
	}	
}

QWidget* vtkPlotter::GetControlWidget(QWidget* parent)
{
	return nullptr;
}

void vtkPlotter::Serialize(ISimpleSerializer *pSerializer)
{
	QtPlotter::Serialize(pSerializer);
	SERIALIZE(render_widgets);
	SERIALIZE(renderer);
}

REGISTERCLASS(vtkPlotter);