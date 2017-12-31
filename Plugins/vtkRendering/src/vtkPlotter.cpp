#include "vtkPlotter.h"
#include "vtkLogRedirect.h"
#include <vtkAxesActor.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkOpenGLRenderWindow.h>

#include <QVTKWidget2.h>

#include "QtOpenGL/QGLContext"
#include <QOpenGLContext>
#include <qgridlayout.h>
#include <qopengl.h>

using namespace EagleLib;
using namespace EagleLib::Plotting;

vtkPlotterBase::vtkPlotterBase() : QtPlotter()
{
    renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkLogRedirect::init();
}

vtkPlotterBase::~vtkPlotterBase()
{
    mo::ThreadSpecificQueue::RemoveFromQueue(this);
    for (auto prop : _auto_remove_props)
    {
        renderer->RemoveViewProp(prop);
    }
}
bool vtkPlotterBase::AcceptsParameter(mo::IParameter* param)
{
    return false;
}

void vtkPlotterBase::SetInput(mo::IParameter* param_)
{
    QtPlotter::SetInput(param_);
}

void vtkPlotterBase::OnParameterUpdate(cv::cuda::Stream* stream)
{
}

void vtkPlotterBase::AddPlot(QWidget* plot_)
{
    auto widget = dynamic_cast<QVTKWidget2*>(plot_);
    if (widget)
    {
        widget->GetRenderWindow()->AddRenderer(renderer);
    }
}

QWidget* vtkPlotterBase::CreatePlot(QWidget* parent)
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
void vtkPlotterBase::PlotInit(bool firstInit)
{
    QtPlotter::PlotInit(firstInit);
    if (firstInit)
    {
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

        this->renderer->AddActor(axes);
    }
}

void vtkPlotterBase::Serialize(ISimpleSerializer* pSerializer)
{
    QtPlotter::Serialize(pSerializer);
    SERIALIZE(render_widgets);
    SERIALIZE(renderer);
}

vtkRenderer* vtkPlotterBase::GetRenderer()
{
    return renderer;
}
void vtkPlotterBase::AddViewProp(vtkProp* prop)
{
    renderer->AddViewProp(prop);
}
void vtkPlotterBase::AddAutoRemoveProp(vtkProp* prop)
{
    renderer->AddViewProp(prop);
    _auto_remove_props.push_back(prop);
}
void vtkPlotterBase::RemoveViewProp(vtkProp* prop)
{
    renderer->RemoveViewProp(prop);
}
void vtkPlotterBase::RenderAll()
{
    for (auto itr : this->render_widgets)
    {
        itr->GetRenderWindow()->Render();
    }
}
// MO_REGISTER_CLASS(vtkPlotterBase);