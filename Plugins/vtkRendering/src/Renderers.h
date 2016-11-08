#pragma once
#include "RuntimeLinkLibrary.h"
#include "EagleLib/plotters/Plotter.h"
#include <EagleLib/rendering/RenderingEngine.h>
#include <EagleLib/utilities/ogl_allocators.h>
#include "vtkPlotter.h"
#include "vtkMatDataBuffer.h"

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkOpenGLTexture.h>
#include <vtkTextureObject.h>
#include <vtkOutputWindow.h>

#include <opencv2/core/opengl.hpp>
#include <boost/thread/mutex.hpp>



SETUP_PROJECT_DEF
class QVTKWidget2;
class vtkPoints;
class vtkFloatArray;
class vtkPolyData;
class vtkPolyDataMapper;

//#define AUTO_BUFFERS

namespace EagleLib
{

    namespace Plotting
    {
        /*struct PLUGIN_EXPORTS vtkImageViewerInfo: public vtkPlotterInfo
        {
            virtual bool AcceptsParameter(mo::Parameter* param);
            virtual std::string GetObjectName();
            virtual std::string GetObjectTooltip();
            virtual std::string GetObjectHelp();
        };*/
        class PLUGIN_EXPORTS vtkImageViewer : public vtkPlotterBase
        {
        public:
            double current_aspect_ratio;
            int texture_stream_index;
            vtkSmartPointer<vtkTextureDataBuffer> textureObject;
            vtkSmartPointer<vtkOpenGLTexture> texture;
            vtkSmartPointer<vtkActor> texturedQuad;
            vtkSmartPointer<vtkPoints> points;
            vtkSmartPointer<vtkFloatArray> textureCoordinates;
            vtkSmartPointer<vtkPolyDataMapper> mapper;
            vtkSmartPointer<vtkPolyData> quad;
            MO_DERIVE(vtkImageViewer, vtkPlotterBase)

            MO_END;
            vtkImageViewer();
            ~vtkImageViewer();
            QWidget* CreatePlot(QWidget* parent);
            virtual void SetInput(mo::IParameter* param_ = nullptr);
            virtual void OnParameterUpdate(cv::cuda::Stream* stream);
            //virtual std::string PlotName() const;
            virtual void Serialize(ISimpleSerializer *pSerializer);
            virtual void Init(bool firstInit);
            static bool AcceptsParameter(mo::IParameter* param);
        };
    }
}
