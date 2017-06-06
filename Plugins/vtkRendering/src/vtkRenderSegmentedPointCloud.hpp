#pragma once
#include "MetaObject/MetaObject.hpp"
#include "EagleLib/Nodes/Node.h"

#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkMatDataBuffer.h"
#include "vtkLODActor.h"
#include "vtkIdTypeArray.h"
#include "vtkPolyData.h"
class vtkRenderWindowInteractor;
class vtkOpenGLRenderWindow;
class vtkOpenGLRenderer;
namespace EagleLib
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS vtkRenderSegmentedPointCloud: public Node
        {
        public:
            void Init(bool firstInit);
            MO_DERIVE(vtkRenderSegmentedPointCloud, Node)
                INPUT(SyncedMemory, input_point_cloud, nullptr);
                OPTIONAL_INPUT(SyncedMemory, class_mask, nullptr);
                PROPERTY(vtkSmartPointer<vtkRenderer>, renderer, vtkSmartPointer<vtkRenderer>());
                PROPERTY(vtkSmartPointer<vtkRenderWindowInteractor>, interactor, vtkSmartPointer<vtkRenderWindowInteractor>());
                PROPERTY(vtkSmartPointer<vtkRenderWindow>, render_window, vtkSmartPointer<vtkRenderWindow>());
                PROPERTY(vtkSmartPointer<vtkMatDataBuffer>, opengl_vbo, vtkSmartPointer<vtkMatDataBuffer>());
                PROPERTY(vtkSmartPointer<vtkPolyData>, polydata, vtkSmartPointer<vtkPolyData>());
                PROPERTY(vtkSmartPointer<vtkIdTypeArray>, initcells, vtkSmartPointer<vtkIdTypeArray>());
                PROPERTY(vtkSmartPointer<vtkLODActor>, actor, vtkSmartPointer<vtkLODActor>());
            MO_END;            
        protected:
            bool processImpl();
        private:
        };
    }
}