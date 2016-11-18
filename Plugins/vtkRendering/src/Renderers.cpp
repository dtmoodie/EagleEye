#include "Renderers.h"
#include "vtkLogRedirect.h"

#include "EagleLib/Nodes/Node.h"
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include "EagleLib/utilities/ObjectPool.hpp"
#include <EagleLib/rcc/SystemTable.hpp>
#include <EagleLib/plotters/PlotInfo.hpp>

#include "vtkTexture.h"
#include "vtkPointData.h"
#include <vtkPolyDataMapper.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkJPEGReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTexture.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkFloatArray.h>
#include <vtkPolygon.h>
#include <vtkGenericOpenGLRenderWindow.h>


using namespace EagleLib;
using namespace EagleLib::Plotting;


bool vtkImageViewer::AcceptsParameter(mo::IParameter* param)
{
    auto type = param->GetTypeInfo();
    if(type == mo::TypeInfo(typeid(cv::cuda::GpuMat)))
    {
        auto typed = dynamic_cast<mo::ITypedParameter<cv::cuda::GpuMat>*>(param);
        auto mat = typed->GetDataPtr();
        if(mat->depth() == CV_8U && (mat->channels() == 1 || mat->channels() == 3 || mat->channels() == 4))
        {
            return true;
        }
    }
    return false;
}


vtkImageViewer::vtkImageViewer():
    vtkPlotterBase()
{
    current_aspect_ratio = 1.0;
    vtkLogRedirect::init();
}
vtkImageViewer::~vtkImageViewer()
{
    mo::ThreadSpecificQueue::RemoveFromQueue(this);
}
void vtkImageViewer::Serialize(ISimpleSerializer *pSerializer)
{
    vtkPlotterBase::Serialize(pSerializer);
    SERIALIZE(texture);
    SERIALIZE(textureObject);
    SERIALIZE(texturedQuad);
    SERIALIZE(textureCoordinates);
    SERIALIZE(points);
    SERIALIZE(mapper);
    SERIALIZE(quad);
}
QWidget* vtkImageViewer::CreatePlot(QWidget* parent)
{
    auto plot = vtkPlotterBase::CreatePlot(parent);
    if(textureObject == nullptr)
    {
        cv::Mat default_texture(cv::Size(100, 100), CV_8UC3, cv::Scalar(255));
        textureObject = vtkSmartPointer<vtkTextureDataBuffer>::New();
        this->textureObject->SetContext(render_widgets.back()->GetRenderWindow());
        texture->SetTextureObject(textureObject);
#ifdef AUTO_BUFFER
        textureObject->image_buffer = PerModuleInterface::GetInstance()->GetSystemTable()->GetSingleton<EagleLib::ogl_allocator>()->get_ogl_buffer(default_texture);
#else
        textureObject->data_buffer.copyFrom(default_texture);
#endif
        textureObject->compile_texture();
    }
    
    return plot;
}
void vtkImageViewer::Init(bool firstInit)
{
    vtkPlotterBase::Init(firstInit);
    if (firstInit)
    {
        texture = vtkSmartPointer<vtkOpenGLTexture>::New();
        // Create a plane
        points = vtkSmartPointer<vtkPoints>::New();
        points->InsertNextPoint(0.0, 0.0, 0.0);
        points->InsertNextPoint(1.0, 0.0, 0.0);
        points->InsertNextPoint(1.0, 1.0, 0.0);
        points->InsertNextPoint(0.0, 1.0, 0.0);
        vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
        polygon->GetPointIds()->SetNumberOfIds(4); //make a quad
        polygon->GetPointIds()->SetId(0, 0);
        polygon->GetPointIds()->SetId(1, 1);
        polygon->GetPointIds()->SetId(2, 2);
        polygon->GetPointIds()->SetId(3, 3);
        polygons->InsertNextCell(polygon);

        quad = vtkSmartPointer<vtkPolyData>::New();
        quad->SetPoints(points);
        quad->SetPolys(polygons);

        textureCoordinates = vtkSmartPointer<vtkFloatArray>::New();
        textureCoordinates->SetNumberOfComponents(3);
        textureCoordinates->SetName("TextureCoordinates");
        float tuple[3];
        tuple[0] =  0.0; tuple[1] =  1.0; tuple[2] = 0.0; textureCoordinates->InsertNextTuple(tuple);
        tuple[0] =  1.0; tuple[1] =  1.0; tuple[2] = 0.0; textureCoordinates->InsertNextTuple(tuple);
        tuple[0] =  1.0; tuple[1] =  0.0; tuple[2] = 0.0; textureCoordinates->InsertNextTuple(tuple);
        tuple[0] =  0.0; tuple[1] =  0.0; tuple[2] = 0.0; textureCoordinates->InsertNextTuple(tuple);
        

        quad->GetPointData()->SetTCoords(textureCoordinates);
        
        mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(quad);

        texturedQuad = vtkSmartPointer<vtkActor>::New();
        texturedQuad->SetMapper(mapper);
        texturedQuad->SetTexture(texture);

        this->renderer->AddActor(texturedQuad);
        this->renderer->ResetCamera();
    }
    texture_stream_index = 0;
}


void vtkImageViewer::SetInput(mo::IParameter* param_)
{
    vtkPlotterBase::SetInput(param_);
}

void vtkImageViewer::OnParameterUpdate(cv::cuda::Stream* stream)
{
    //rmt_ScopedCPUSample(vtkImageViewer_OnParameterUpdate);
    if (stream)
    {
        bool shown = false;
        for (auto itr : render_widgets)
        {
            if (itr->isVisible())
                shown = true;

        }
        if (shown == false)
            return;

        cv::cuda::GpuMat d_mat = *(dynamic_cast<mo::ITypedParameter<cv::cuda::GpuMat>*>(parameter)->GetDataPtr());
        
        mo::ThreadSpecificQueue::Push(std::bind<void>([d_mat, stream, this]()->void
        {
            {
                //rmt_ScopedCPUSample(opengl_buffer_fill);
                // Need to adjust points to the aspect ratio of the input image
                double aspect_ratio = (double)d_mat.cols / (double)d_mat.rows;
                if(aspect_ratio != current_aspect_ratio)
                {
                    points->SetPoint(0, 0.0         ,  0.0, 0.0);
                    points->SetPoint(1, aspect_ratio,  0.0, 0.0);
                    points->SetPoint(2, aspect_ratio,  1.0, 0.0);
                    points->SetPoint(3, 0.0         ,  1.0, 0.0);
                    points->Modified();
                    current_aspect_ratio = aspect_ratio;
                }
#ifdef AUTO_BUFFERS
                textureObject->image_buffer = PerModuleInterface::GetInstance()->GetSystemTable()->GetSingleton<EagleLib::ogl_allocator>()->get_ogl_buffer(d_mat, *stream);
#else
                textureObject->data_buffer.copyFrom(d_mat, *stream);
#endif

                stream->waitForCompletion();
            }

            //std::lock_guard<std::recursive_mutex> lock(this->mtx());
            {
                //rmt_ScopedCPUSample(texture_creation);

                textureObject->compile_texture();
            }
            {
                //rmt_ScopedCPUSample(Rendering);
                for (auto itr : this->render_widgets)
                {
                    itr->GetRenderWindow()->Render();
                }
            }
        }), mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI), this);
    }
}


MO_REGISTER_CLASS(vtkImageViewer);
