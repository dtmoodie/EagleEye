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
        /*struct PLUGIN_EXPORTS vtkPointCloudPlotterInfo: public PlotterInfo
        {
            
            virtual bool AcceptsParameter(mo::IParameter* param);
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
        };*/
        PLUGIN_EXPORTS void convertPointCloudToVTKPolyData(
            cv::InputArray cloud,
            vtkSmartPointer<vtkPolyData> &polydata,
            vtkSmartPointer<vtkIdTypeArray> &initcells, cv::cuda::Stream& stream);

        PLUGIN_EXPORTS void createActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet> &data,
                vtkSmartPointer<vtkLODActor> &actor,
                bool use_scalars);

        PLUGIN_EXPORTS void updateCells(
                vtkSmartPointer<vtkIdTypeArray> &cells,
                vtkSmartPointer<vtkIdTypeArray> &initcells,
                vtkIdType nr_points);

        class PLUGIN_EXPORTS vtkPointCloudPlotter: public vtkPlotterBase
        {
        public:
            MO_DERIVE(vtkPointCloudPlotter, vtkPlotterBase)
            MO_END;
            ~vtkPointCloudPlotter();
            static bool AcceptsParameter(mo::IParameter* param);
            virtual void SetInput(mo::IParameter* param_ = nullptr);
            virtual void OnParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnGpuMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnSyncedMemUpdate(cv::cuda::Stream* stream);

        private:
            // Point clouds get uploaded / copied into this opengl vertex buffer for rendering
            vtkSmartPointer<vtkMatDataBuffer> _opengl_vbo;
            vtkSmartPointer<vtkPolyData> polydata;
            vtkSmartPointer<vtkIdTypeArray> initcells;
            vtkSmartPointer<vtkLODActor> actor;
        };


    }
}