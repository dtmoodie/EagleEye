#pragma once

#include "vtkPlotter.h"
#include "vtkMatDataBuffer.h"

#include <vtkPolyData.h>
#include <vtkIdTypeArray.h>
#include <vtkLODActor.h>

namespace EagleLib
{
    namespace Plotting
    {
        struct PLUGIN_EXPORTS vtkPointCloudPlotterInfo: public PlotterInfo
        {
            virtual Plotter::PlotterType GetPlotType();
            virtual bool AcceptsParameter(Parameters::Parameter* param);
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
        };
        class PLUGIN_EXPORTS vtkPointCloudPlotter: public vtkPlotter
        {
        public:
            ~vtkPointCloudPlotter();
            virtual bool AcceptsParameter(Parameters::Parameter* param);
            virtual void SetInput(Parameters::Parameter* param_ = nullptr);
            virtual void OnParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnGpuMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnSyncedMemUpdate(cv::cuda::Stream* stream);
            virtual std::string PlotName() const;

        private:
            // Point clouds get uploaded / copied into this opengl vertex buffer for rendering
            vtkSmartPointer<vtkMatDataBuffer> _opengl_vbo;
            vtkSmartPointer<vtkPolyData> polydata;
            vtkSmartPointer<vtkIdTypeArray> initcells;
            vtkSmartPointer<vtkLODActor> actor;
        };


    }
}