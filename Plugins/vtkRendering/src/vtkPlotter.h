#pragma once
#include "EagleLib/plotters/Plotter.h"
#include "vtkSmartPointer.h"
#include "vtkRenderer.h"
#include "QVTKWidget2.h"

#define VTK_VERSION_ "7.1"
#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionStyle-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionWidgets-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersExtraction-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonDataModel-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonMath-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonCore-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtksys-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonMisc-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonSystem-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonTransforms-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonExecutionModel-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersCore-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersGeneral-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonComputationalGeometry-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersStatistics-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingFourier-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingCore-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkalglib-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersSources-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingCore-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonColor-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersGeometry-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingLOD-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersModeling-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOPLY-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOGeometry-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOCore-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkzlib-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersTexture-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingFreeType-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkfreetype-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOExport-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOImage-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkDICOMParser-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkmetaio-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkjpeg-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkpng-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtktiff-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingAnnotation-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingColor-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingContext2D-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingLabel-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingOpenGL2-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingHybrid-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkglew-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingQt-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkGUISupportQt-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkGUISupportQtOpenGL-" VTK_VERSION_ "d.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgetsd.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Guid.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Cored.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5OpenGLd.lib");
RUNTIME_COMPILER_LINKLIBRARY("OpenGL32.lib")

#else
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionStyle-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionWidgets-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersExtraction-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonDataModel-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonMath-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonCore-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtksys-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonMisc-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonSystem-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonTransforms-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonExecutionModel-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersCore-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersGeneral-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonComputationalGeometry-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersStatistics-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingFourier-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingCore-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkalglib-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersSources-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingCore-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkCommonColor-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersGeometry-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingLOD-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersModeling-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOPLY-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOGeometry-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOCore-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkzlib-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkFiltersTexture-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingFreeType-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkfreetype-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOExport-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkIOImage-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkDICOMParser-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkmetaio-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkjpeg-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkpng-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtktiff-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingAnnotation-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingColor-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingContext2D-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingLabel-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingOpenGL2-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkImagingHybrid-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkglew-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkRenderingQt-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkGUISupportQt-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("vtkGUISupportQtOpenGL-" VTK_VERSION_ ".lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgets.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Gui.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Core.lib")
#endif

#else
#endif
namespace EagleLib
{
    namespace Plotting
    {
        struct PLUGIN_EXPORTS vtkPlotterInfo: public PlotterInfo
        {
            virtual Plotter::PlotterType GetPlotType();
        };
        class PLUGIN_EXPORTS vtkPlotter : public QtPlotter
        {
            std::list<vtkProp*> _auto_remove_props;
        protected:
            vtkSmartPointer<vtkRenderer> renderer;
            std::list<QVTKWidget2*> render_widgets;
            
            vtkRenderer* GetRenderer();
            void AddViewProp(vtkProp* prop);
            void AddAutoRemoveProp(vtkProp* prop);
            void RemoveViewProp(vtkProp* prop);
            void RenderAll();
        public:
            vtkPlotter();
            virtual ~vtkPlotter();
            virtual bool AcceptsParameter(Parameters::Parameter* param);
            virtual void SetInput(Parameters::Parameter* param_ = nullptr);
            virtual void OnParameterUpdate(cv::cuda::Stream* stream);
            virtual std::string PlotName() const;
            virtual void PlotInit(bool firstInit);
            virtual void AddPlot(QWidget* plot_);

            virtual QWidget* CreatePlot(QWidget* parent);

            virtual void Serialize(ISimpleSerializer *pSerializer);
        };
    }
}