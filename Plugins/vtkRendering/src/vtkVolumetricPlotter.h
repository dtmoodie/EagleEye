#pragma once
#define Qt5_FOUND
#define HAVE_OPENCV
#define PARAMETERS_GENERATE_UI

#include "vtkPlotter.h"
#include "vtkMatDataBuffer.h"
#include "MetaObject/Parameters/Types.hpp"

#include <vtkPolyData.h>
#include <vtkIdTypeArray.h>
#include <vtkLODActor.h>




#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolume-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolumeOpenGL2-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionWidgets-" VTK_VERSION_ "d.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolume-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingVolumeOpenGL2-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionWidgets-" VTK_VERSION_ ".lib");
#endif
#else

#endif

class vtkSmartVolumeMapper;
class vtkVolumeProperty;
class vtkPiecewiseFunction;
class vtkColorTransferFunction;
class vtkClipVolume;
class vtkBox;
class vtkBoxWidget;
class vtkBoxWidgetCallback;
namespace EagleLib
{
    namespace Plotting
    {
        class PLUGIN_EXPORTS vtkVolumetricPlotter : public vtkPlotterBase
        {
        public:
            vtkVolumetricPlotter();
            virtual ~vtkVolumetricPlotter();
            static bool AcceptsParameter(mo::IParameter* param);
            virtual void SetInput(mo::IParameter* param_ = nullptr);
            virtual void OnParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnGpuMatParameterUpdate(cv::cuda::Stream* stream);
            virtual void OnSyncedMemUpdate(cv::cuda::Stream* stream);
            virtual std::string PlotName() const;
            virtual void onUpdate(mo::IParameter* param = nullptr, cv::cuda::Stream* stream = nullptr);
            virtual void PlotInit(bool firstInit);
            
            MO_DERIVE(vtkVolumetricPlotter, vtkPlotterBase);
                PARAM(cv::Rect, region_of_interest, cv::Rect(0,0, 10, 10));
                PARAM(float, opacity_max_value, 2048 );
                PARAM(float, opacity_min_value, 50 );
                PARAM(float, opacity_sharpness, 0.5);
                PARAM(mo::EnumParameter, colormapping_scheme, mo::EnumParameter());
            MO_END;
        protected:
            vtkSmartPointer<vtkVolume> _volume;
            vtkSmartPointer<vtkSmartVolumeMapper> _mapper;
            vtkSmartPointer<vtkVolumeProperty> _volumeProperty;
            vtkSmartPointer<vtkPiecewiseFunction> _compositeOpacity;
            vtkSmartPointer<vtkColorTransferFunction> _color;
            vtkSmartPointer<vtkClipVolume> _clipping_volume;
            vtkSmartPointer<vtkBoxWidget> _clipping_function;
            vtkBoxWidgetCallback* _callback;
        };
    }
}