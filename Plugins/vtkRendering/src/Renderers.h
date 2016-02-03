#pragma once
#include "RuntimeLinkLibrary.h"
#include <EagleLib/Defs.hpp>
#include <EagleLib/Project_defs.hpp>
#include "EagleLib/plotters/Plotter.h"


#include <EagleLib/rendering/RenderingEngine.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkOpenGLTexture.h>
#include <opencv2/core/opengl.hpp>
#include <boost/thread/mutex.hpp>
#include <vtkTextureObject.h>
#include <EagleLib/utilities/ogl_allocators.h>
#define VTK_VERSION_ "7.1"
#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionStyle-" VTK_VERSION_ "d.lib");
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
RUNTIME_COMPILER_LINKLIBRARY("libParameterd.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("libParameter.lib")
RUNTIME_COMPILER_LINKLIBRARY("vtkInteractionStyle-" VTK_VERSION_ ".lib");
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
SETUP_PROJECT_DEF
class QVTKWidget2;
class vtkPoints;
class vtkFloatArray;
class vtkPolyData;
class vtkPolyDataMapper;
namespace EagleLib
{
	class vtkOpenGLCudaImage : public vtkTextureObject
	{
	public:
		static vtkOpenGLCudaImage* New();
		vtkTypeMacro(vtkOpenGLCudaImage, vtkTextureObject);
		
        void compile_texture();
        EagleLib::pool::Ptr<cv::ogl::Buffer> image_buffer;
		//cv::ogl::Buffer image_buffer;
		virtual void Bind();
		boost::mutex mtx;
	private:
		vtkOpenGLCudaImage();
		
	};

	class vtkPlotter : public QtPlotter
	{
	protected:
		std::list<QVTKWidget2*> render_widgets;
		vtkSmartPointer<vtkRenderer> renderer;
	public:
		vtkPlotter();
		virtual bool AcceptsParameter(Parameters::Parameter::Ptr param);
		virtual void SetInput(Parameters::Parameter::Ptr param_ = Parameters::Parameter::Ptr());
		virtual void OnParameterUpdate(cv::cuda::Stream* stream);
		virtual std::string PlotName() const;

		virtual void AddPlot(QWidget* plot_);

		virtual QWidget* CreatePlot(QWidget* parent);
		virtual QWidget* GetControlWidget(QWidget* parent);
		
		virtual void Serialize(ISimpleSerializer *pSerializer);
		virtual void Init(bool firstInit);
	};

	class vtkImageViewer : public vtkPlotter
	{
	public:
        double current_aspect_ratio;
		int texture_stream_index;
		vtkSmartPointer<vtkOpenGLCudaImage> textureObject;
		vtkSmartPointer<vtkOpenGLTexture> texture;
        vtkSmartPointer<vtkActor> texturedQuad;
        vtkSmartPointer<vtkPoints> points;
        vtkSmartPointer<vtkFloatArray> textureCoordinates;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkPolyData> quad;

		vtkImageViewer();
        QWidget* CreatePlot(QWidget* parent);
		virtual bool AcceptsParameter(Parameters::Parameter::Ptr param);
		virtual void SetInput(Parameters::Parameter::Ptr param_ = Parameters::Parameter::Ptr());
		virtual void OnParameterUpdate(cv::cuda::Stream* stream);
		virtual std::string PlotName() const;
		virtual void Serialize(ISimpleSerializer *pSerializer);
		virtual void Init(bool firstInit);
	};
}
