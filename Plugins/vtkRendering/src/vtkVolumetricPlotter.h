#pragma once
#include "vtkPlotter.h"
#include "vtkMatDataBuffer.h"

#include <vtkPolyData.h>
#include <vtkIdTypeArray.h>
#include <vtkLODActor.h>
#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolume-" VTK_VERSION_ "d.lib")
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolumeOpenGL2-" VTK_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolume-" VTK_VERSION_ ".lib")
#endif
#else

#endif
namespace EagleLib
{
    namespace Plotting
    {
        struct PLUGIN_EXPORTS vtkVolumetricPlotterInfo: public vtkPlotterInfo
        {
            virtual bool AcceptsParameter(Parameters::Parameter* param);
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
        };

        class PLUGIN_EXPORTS vtkVolumetricPlotter : public vtkPlotter
        {
        public:
            virtual ~vtkVolumetricPlotter();
            virtual bool AcceptsParameter(Parameters::Parameter* param);
            virtual void SetInput(Parameters::Parameter* param_ = nullptr);
            virtual void OnParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnGpuMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnSyncedMemUpdate(cv::cuda::Stream* stream);
            virtual std::string PlotName() const;
        };
    }
}