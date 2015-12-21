#include "Renderers.h"
#include "nodes/Node.h"
#include "Manager.h"

SETUP_PROJECT_IMPL
using namespace EagleLib;

vtkRenderEngine::vtkRenderEngine()
{
	renderer =	vtkSmartPointer<vtkRenderer>::New();
	renderWindow =	vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
}

void vtkRenderEngine::Render()
{

}

void vtkRenderEngine::AddRenderScene(std::shared_ptr<IRenderScene> scene)
{

}

REGISTERCLASS(vtkRenderEngine);

