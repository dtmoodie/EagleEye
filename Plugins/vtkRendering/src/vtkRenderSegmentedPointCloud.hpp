#pragma once
#include "MetaObject/MetaObject.hpp"
#include "EagleLib/Nodes/Node.h"

#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkMatDataBuffer.h"
#include "vtkLODActor.h"
#include "vtkIdTypeArray.h"
#include "vtkPolyData.h"

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
                INPUT(SyncedMemory, class_mask, nullptr);
                PROPERTY(vtkSmartPointer<vtkRenderer>, renderer, vtkSmartPointer<vtkRenderer>::New());
                PROPERTY(vtkSmartPointer<vtkMatDataBuffer>, opengl_vbo, vtkSmartPointer<vtkMatDataBuffer>::New());
                PROPERTY(vtkSmartPointer<vtkPolyData>, polydata, vtkSmartPointer<vtkPolyData>::New());
                PROPERTY(vtkSmartPointer<vtkIdTypeArray>, initcells, vtkSmartPointer<vtkIdTypeArray>::New());
                PROPERTY(vtkSmartPointer<vtkLODActor>, actor, vtkSmartPointer<vtkLODActor>());
            MO_END;            
        protected:
            bool ProcessImpl();
        private:
        };
    }
}