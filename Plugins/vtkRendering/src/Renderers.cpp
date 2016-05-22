#include "Renderers.h"
#include "Remotery.h"
#include "EagleLib/nodes/Node.h"
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include "EagleLib/utilities/ObjectPool.hpp"
#include <EagleLib/rcc/SystemTable.hpp>

#include "parameters/UI/InterThread.hpp"

#include <QVTKWidget2.h>
#include <qopengl.h>
#include "QtOpenGL/QGLContext"
#include <QOpenGLContext>

#include "vtkTexture.h"
#include "vtkPointData.h"
#include <vtkAxesActor.h>
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
#include <parameters/ParameteredObjectImpl.hpp>


SETUP_PROJECT_IMPL
using namespace EagleLib;

/*
vtkRenderEngine::vtkRenderEngine()
{
	renderer =	vtkSmartPointer<vtkRenderer>::New();
	renderWindow =	vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
}

void vtkRenderEngine::Render()
{

}

void vtkRenderEngine::AddRenderScene(std::shared_ptr<IRenderScene> scene)
{

}

REGISTERCLASS(vtkRenderEngine);
*/

vtkPlotter::vtkPlotter()
{
	renderer = vtkSmartPointer<vtkRenderer>::New();
}

bool vtkPlotter::AcceptsParameter(Parameters::Parameter::Ptr param)
{
	return false;
}
std::string vtkPlotter::PlotName() const
{
	return "vtkPlotter";
}

void vtkPlotter::SetInput(Parameters::Parameter::Ptr param_)
{
	QtPlotter::SetInput(param_);
}

void vtkPlotter::OnParameterUpdate(cv::cuda::Stream* stream)
{

}

void vtkPlotter::AddPlot(QWidget* plot_)
{
	auto widget = dynamic_cast<QVTKWidget2*>(plot_);
	if (widget)
	{
		widget->GetRenderWindow()->AddRenderer(renderer);
	}
}

QWidget* vtkPlotter::CreatePlot(QWidget* parent)
{
	QOpenGLContext* draw_context = new QOpenGLContext();

#if QT_VERSION > 0x050500
    auto global_context = QOpenGLContext::globalShareContext();
    draw_context->setShareContext(global_context);
#endif
	auto widget = new QVTKWidget2(QGLContext::fromOpenGLContext(draw_context), parent);
    widget->makeCurrent();
	widget->show();
	widget->setMinimumWidth(100);
	widget->setMinimumHeight(100);
	render_widgets.push_back(widget);
    widget->GetRenderWindow()->OpenGLInit();
	widget->GetRenderWindow()->AddRenderer(renderer);
	
	return widget;
}

QWidget* vtkPlotter::GetControlWidget(QWidget* parent)
{
	return nullptr;
}

void vtkPlotter::Serialize(ISimpleSerializer *pSerializer)
{
	QtPlotter::Serialize(pSerializer);
	SERIALIZE(render_widgets);
	SERIALIZE(renderer);
}



REGISTERCLASS(vtkPlotter);

vtkOpenGLCudaImage::vtkOpenGLCudaImage():
	vtkTextureObject()
{
	
}
vtkOpenGLCudaImage* vtkOpenGLCudaImage::New()
{
	return new vtkOpenGLCudaImage();
}

void vtkOpenGLCudaImage::compile_texture()
{
    try
    {
        if(image_buffer)
        {
            image_buffer->bind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
            
            if ((this->Width != image_buffer->cols() || this->Height != image_buffer->rows() || this->Components != image_buffer->channels()) && Context)
            {
                InternalFormat = GL_RGB8;
                int vtk_type = 0;
                switch(image_buffer->depth())
                {
                    case CV_8U: vtk_type = VTK_UNSIGNED_CHAR; break;
                    case CV_16U: vtk_type = VTK_UNSIGNED_SHORT; break;
                    case CV_32S: vtk_type = VTK_INT; break;
                    case CV_32F: vtk_type = VTK_FLOAT; break;
                    case CV_64F: vtk_type = VTK_DOUBLE; break;
                }
                Allocate2D(image_buffer->cols(), image_buffer->rows(), image_buffer->channels(), vtk_type);
            }
            else
            {
                this->Activate();
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, this->Format, this->Type, NULL);
            }
            this->Deactivate();
            image_buffer->unbind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
            Modified();
        }
        
    }
    catch (cv::Exception& e)
    {
        LOG(error) << e.what();
    }
    catch (...)
    {

    }
}
void vtkOpenGLCudaImage::Bind()
{
	vtkTextureObject::Bind();
}


vtkImageViewer::vtkImageViewer():
	vtkPlotter()
{
    current_aspect_ratio = 1.0;
}
void vtkImageViewer::Serialize(ISimpleSerializer *pSerializer)
{
	vtkPlotter::Serialize(pSerializer);
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
    auto plot = vtkPlotter::CreatePlot(parent);
    if(textureObject == nullptr)
    {
        cv::Mat default_texture(cv::Size(100, 100), CV_8UC3, cv::Scalar(255));
        textureObject = vtkSmartPointer<vtkOpenGLCudaImage>::New();
        this->textureObject->SetContext(render_widgets.back()->GetRenderWindow());
        texture->SetTextureObject(textureObject);
        textureObject->image_buffer = PerModuleInterface::GetInstance()->GetSystemTable()->GetSingleton<EagleLib::ogl_allocator>()->get_ogl_buffer(default_texture);
        textureObject->compile_texture();
        //textureObject->image_buffer.copyFrom(default_texture);
    }
    
    return plot;
}
void vtkImageViewer::Init(bool firstInit)
{
	vtkPlotter::Init(firstInit);
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
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

        this->renderer->AddActor(axes);
		this->renderer->AddActor(texturedQuad);
		this->renderer->ResetCamera();
	}
	texture_stream_index = 0;
}
bool vtkImageViewer::AcceptsParameter(Parameters::Parameter::Ptr param)
{
	if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
	{
		return true;
	}
	if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::Mat)))
	{
		return false;
	}
	if (param->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::HostMem)))
	{
		return false;
	}
	return false;
}

void vtkImageViewer::SetInput(Parameters::Parameter::Ptr param_)
{
	vtkPlotter::SetInput(param_);
}

void vtkImageViewer::OnParameterUpdate(cv::cuda::Stream* stream)
{
	rmt_ScopedCPUSample(vtkImageViewer_OnParameterUpdate);
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

		cv::cuda::GpuMat d_mat = *std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param)->Data();
		
		
		Parameters::UI::UiCallbackService::Instance()->post(boost::bind<void>([d_mat, stream, this]()->void
		{
			{
				rmt_ScopedCPUSample(opengl_buffer_fill);
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
                textureObject->image_buffer = PerModuleInterface::GetInstance()->GetSystemTable()->GetSingleton<EagleLib::ogl_allocator>()->get_ogl_buffer(d_mat, *stream);
                
				stream->waitForCompletion();
			}
			
			std::lock_guard<std::recursive_mutex> lock(this->mtx);
			{
				rmt_ScopedCPUSample(texture_creation);
                
                textureObject->compile_texture();
			}
			{
				rmt_ScopedCPUSample(Rendering);
				for (auto itr : this->render_widgets)
				{
					itr->GetRenderWindow()->Render();
				}
			}
		}));
		/*EagleLib::cuda::enqueue_callback_async(
			[this, d_mat, stream, current_texture]()->void
		{
			Parameters::UI::UiCallbackService::Instance()->post(
				boost::bind<void>([d_mat, this, stream, current_texture]()->void
			{
				{
					boost::mutex::scoped_lock lock(current_texture->mtx);
					rmt_ScopedCPUSample(Gpu_to_ogl_buffer);

					// Needs to be done on the UI thread because I can't bind a buffer without a valid opengl context which only exists on the UI thread.
					current_texture->image_buffer.copyFrom(d_mat, *stream, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);

					EagleLib::cuda::enqueue_callback_async([this, stream, d_mat, current_texture]()->void
					{
						Parameters::UI::UiCallbackService::Instance()->post(boost::bind<void>([this, d_mat, current_texture]()->void
						{
							boost::recursive_mutex::scoped_lock lock(this->mtx);
							{
								rmt_ScopedCPUSample(texture_creation);
								current_texture->map_gpu_mat(d_mat);
								texture->SetTextureObject(current_texture);
							}
							{
								rmt_ScopedCPUSample(Rendering);
								for (auto itr : this->render_widgets)
								{
									itr->GetRenderWindow()->Render();
								}
							}
						}));
					}, *stream);
				}
			}));
		}, *stream);*/
	}
}

std::string vtkImageViewer::PlotName() const
{
	return "vtkImageViewer";
}

REGISTERCLASS(vtkImageViewer);
