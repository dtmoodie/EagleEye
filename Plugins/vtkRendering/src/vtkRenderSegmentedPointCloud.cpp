#include "EagleLib/Nodes/NodeInfo.hpp"
#include "vtkRenderSegmentedPointCloud.hpp"
#include "vtkAxesActor.h"
#include "vtkPointCloudPlotter.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include <vtkOpenGLRenderer.h>
#include <vtkOpenGLRenderWindow.h>

using namespace EagleLib;
using namespace EagleLib::Nodes;


void vtkRenderSegmentedPointCloud::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
        renderer = vtkOpenGLRenderer::New();
        render_window = vtkOpenGLRenderWindow::New();
        interactor = vtkRenderWindowInteractor::New();
        this->renderer->AddActor(axes);
        this->render_window->AddRenderer(renderer);
        this->interactor->SetRenderWindow(render_window);
        this->renderer->SetBackground(0.3, 0.6, 0.3);
        this->interactor->Initialize();
        
    }
}

bool vtkRenderSegmentedPointCloud::ProcessImpl()
{
    //opengl_vbo->data_buffer.copyFrom(input_point_cloud->GetGpuMat(Stream()), Stream());
    EagleLib::Plotting::convertPointCloudToVTKPolyData(input_point_cloud->GetMat(Stream()), 
        polydata, initcells, Stream());
    bool new_actor = actor == nullptr;
    Plotting::createActorFromVTKDataSet(polydata, actor, false);
    if(new_actor)
        renderer->AddActor(actor);
    actor->Modified();
    render_window->Render();
    this->interactor->Start();
    return true;
}

MO_REGISTER_CLASS(vtkRenderSegmentedPointCloud);