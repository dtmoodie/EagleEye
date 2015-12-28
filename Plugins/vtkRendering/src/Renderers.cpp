#include "Renderers.h"
#include "nodes/Node.h"
#include "Manager.h"
//#include "QVTKWidget.h"
#include <QVTKWidget2.h>
#include <qopengl.h>
#include "QtOpenGL/QGLContext"
#include "vtkTexture.h"
#include "vtkPointData.h"
#include <vtkPolyDataMapper.h>
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include "UI/InterThread.hpp"
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
#include "Remotery.h"
#include "EagleLib/utilities/ObjectPool.hpp"
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
	auto context = new QGLContext(QGLFormat());
	//auto widget = new QVTKWidget2(new QGLContext(QGLFormat()), parent);
	auto widget = new QVTKWidget2(context, parent);
	//widget->GetRenderWindow()->OpenGLInit();
	//widget->GetRenderWindow()->InitializeTextureInternalFormats();
	widget->show();
	widget->setMinimumWidth(100);
	widget->setMinimumHeight(100);
	render_widgets.push_back(widget);
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

void vtkPlotter::Init(bool firstInit)
{
	QtPlotter::Init(firstInit);
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

void vtkOpenGLCudaImage::map_gpu_mat(cv::cuda::GpuMat image)
{
	try
	{
		CV_Assert(image.depth() == CV_8U);
		image_buffer.bind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
		if (this->Width != image.cols || this->Height != image.rows || this->Components != image.channels() && Context)
		{
			InternalFormat = GL_RGB8;
			Allocate2D(image.cols, image.rows, image.channels(), VTK_UNSIGNED_CHAR);
		}
		else
		{
			this->Activate();
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, GL_BGR, GL_UNSIGNED_BYTE, NULL);
			//glTexImage2D(this->Target, 0, static_cast<GLint>(this->InternalFormat),
			//	static_cast<GLsizei>(this->Width),
			//	static_cast<GLsizei>(this->Height),
			//	0, this->Format, this->Type, 0);
		}
		//glFinish();
		image_buffer.unbind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
		Modified();
	}
	catch (cv::Exception& e)
	{
		BOOST_LOG_TRIVIAL(error) << e.what();
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
	
}
void vtkImageViewer::Serialize(ISimpleSerializer *pSerializer)
{
	vtkPlotter::Serialize(pSerializer);
	SERIALIZE(texture);
	SERIALIZE(textureObject);
}
void vtkImageViewer::Init(bool firstInit)
{
	vtkPlotter::Init(firstInit);
	if (firstInit)
	{
		texture = vtkSmartPointer<vtkOpenGLTexture>::New();
		// Create a plane
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
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

		vtkSmartPointer<vtkPolyData> quad = vtkSmartPointer<vtkPolyData>::New();
		quad->SetPoints(points);
		quad->SetPolys(polygons);

		vtkSmartPointer<vtkFloatArray> textureCoordinates = vtkSmartPointer<vtkFloatArray>::New();
		textureCoordinates->SetNumberOfComponents(3);
		textureCoordinates->SetName("TextureCoordinates");
		float tuple[3] = { 0.0, 1.0, 0.0 };
		textureCoordinates->InsertNextTuple(tuple);
		tuple[0] = 1.0; tuple[1] = 1.0; tuple[2] = 0.0;
		textureCoordinates->InsertNextTuple(tuple);
		tuple[0] = 1.0; tuple[1] = 0.0; tuple[2] = 0.0;
		textureCoordinates->InsertNextTuple(tuple);
		tuple[0] = 0.0; tuple[1] = 0.0; tuple[2] = 0.0;
		textureCoordinates->InsertNextTuple(tuple);

		quad->GetPointData()->SetTCoords(textureCoordinates);

		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputData(quad);

		vtkSmartPointer<vtkActor> texturedQuad = vtkSmartPointer<vtkActor>::New();
		texturedQuad->SetMapper(mapper);
		texturedQuad->SetTexture(texture);

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
		cv::cuda::GpuMat d_mat = *std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param)->Data();
		//texture_stream_index = (texture_stream_index + 1) % 2;
		//auto& current_texture = textureObject[texture_stream_index];
		auto& current_texture = textureObject;
		if (current_texture == nullptr)
		{
			boost::recursive_mutex::scoped_lock lock(this->mtx);
			current_texture = vtkSmartPointer<vtkOpenGLCudaImage>::New();
			current_texture->SetContext((*render_widgets.begin())->GetRenderWindow());
		}

		Parameters::UI::UiCallbackService::Instance()->post(boost::bind<void>([current_texture, d_mat, stream, this]()->void
		{
			{
				rmt_ScopedCPUSample(opengl_buffer_fill);
				current_texture->image_buffer.copyFrom(d_mat, *stream, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
				stream->waitForCompletion();
			}
			
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