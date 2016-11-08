#include "EagleLib/Nodes/NodeInfo.hpp"
#include "vtkRenderSegmentedPointCloud.hpp"
#include "vtkAxesActor.h"
#include "vtkPointCloudPlotter.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;


void vtkRenderSegmentedPointCloud::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

        this->renderer->AddActor(axes);
    }
}

bool vtkRenderSegmentedPointCloud::ProcessImpl()
{
    //opengl_vbo->data_buffer.copyFrom(input_point_cloud->GetGpuMat(Stream()), Stream());
    EagleLib::Plotting::convertPointCloudToVTKPolyData(input_point_cloud->GetGpuMat(Stream()), 
        polydata, initcells, Stream());
    bool new_actor = actor == nullptr;
    Plotting::createActorFromVTKDataSet(polydata, actor, false);
    if(new_actor)
        renderer->AddViewProp(actor);
    actor->Modified();
    this->renderer->Render();
    return true;
}

MO_REGISTER_CLASS(vtkRenderSegmentedPointCloud);