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
	//boost::mutex::scoped_lock lock(mtx);
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
			glTexImage2D(this->Target, 0, static_cast<GLint>(this->InternalFormat),
				static_cast<GLsizei>(this->Width),
				static_cast<GLsizei>(this->Height),
				0, this->Format, this->Type, 0);
			
			//vtkOpenGLCheckErrorMacro("failed at glTexImage2D");
			this->Deactivate();
		}
		image_buffer.unbind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
		/*
		image_buffer.copyFrom(image);
		//this->Index = image_buffer.texId();
		//this->LoadTime.Modified();
		this->Width = image.cols;
		this->Height = image.rows;
		this->Components = image.channels();
		this->Depth = 1;
		this->NumberOfDimensions = 2;
		this->Target = GL_TEXTURE_2D;
		if (Context)
		{
			this->GetDataType(VTK_UNSIGNED_CHAR);
			this->GetInternalFormat(VTK_UNSIGNED_CHAR, Components, false);
			this->GetFormat(VTK_UNSIGNED_CHAR, Components, false);
		}		
		Modified();
		*/

	}
	catch (cv::Exception& e)
	{
		BOOST_LOG_TRIVIAL(error) << e.what();
	}
	catch (...)
	{

	}
}
//void vtkOpenGLCudaImage::Bind()
//{
	//image_buffer.bind();
	//if (this->AutoParameters && (this->GetMTime() > this->SendParametersTime))
	//{
//		this->SendParameters();
	//}
//}


vtkImageViewer::vtkImageViewer():
	vtkPlotter()
{
	texture = vtkSmartPointer<vtkOpenGLTexture>::New();
	//textureObject = vtkSmartPointer<vtkOpenGLCudaImage>::New();
	//texture->SetTextureObject(textureObject);
	//vtkSmartPointer<vtkJPEGReader> jPEGReader =	vtkSmartPointer<vtkJPEGReader>::New();
	//jPEGReader->SetFileName("C:/Users/Public/Pictures/Sample Pictures/Koala.jpg");
	//texture = vtkSmartPointer<vtkTexture>::New();
	//texture->SetInputConnection(jPEGReader->GetOutputPort());

	// Create a plane
	vtkSmartPointer<vtkPoints> points =	vtkSmartPointer<vtkPoints>::New();
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

	vtkSmartPointer<vtkPolyData> quad =	vtkSmartPointer<vtkPolyData>::New();
		quad->SetPoints(points);
		quad->SetPolys(polygons);

	vtkSmartPointer<vtkFloatArray> textureCoordinates =	vtkSmartPointer<vtkFloatArray>::New();
		textureCoordinates->SetNumberOfComponents(3);
		textureCoordinates->SetName("TextureCoordinates");
		float tuple[3] = { 0.0, 0.0, 0.0 };
		textureCoordinates->InsertNextTuple(tuple);
		tuple[0] = 1.0; tuple[1] = 0.0; tuple[2] = 0.0;
		textureCoordinates->InsertNextTuple(tuple);
		tuple[0] = 1.0; tuple[1] = 1.0; tuple[2] = 0.0;
		textureCoordinates->InsertNextTuple(tuple);
		tuple[0] = 0.0; tuple[1] = 1.0; tuple[2] = 0.0;
		textureCoordinates->InsertNextTuple(tuple);

	quad->GetPointData()->SetTCoords(textureCoordinates);

	vtkSmartPointer<vtkPolyDataMapper> mapper =	vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputData(quad);

	vtkSmartPointer<vtkActor> texturedQuad = vtkSmartPointer<vtkActor>::New();
		texturedQuad->SetMapper(mapper);
		texturedQuad->SetTexture(texture);

	this->renderer->AddActor(texturedQuad);
	this->renderer->ResetCamera();
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
	if (param_->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
	{
		param = param_;
		RegisterParameterCallback(param.get(), boost::bind(&vtkImageViewer::OnParameterUpdate, this, _1), true, true);
		//param->RegisterNotifier(boost::bind(&vtkImageViewer::OnParameterUpdate, this, _1));
	}
	if (param_->GetTypeInfo() == Loki::TypeInfo(typeid(cv::Mat)))
	{
		
	}
	if (param_->GetTypeInfo() == Loki::TypeInfo(typeid(cv::cuda::HostMem)))
	{
		
	}
}

void vtkImageViewer::OnParameterUpdate(cv::cuda::Stream* stream)
{
	if (stream)
	{
		cv::cuda::GpuMat d_mat = *std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param)->Data();
		if (textureObject == nullptr)
		{
			//texture = vtkSmartPointer<vtkOpenGLTexture>::New();
			textureObject = vtkSmartPointer<vtkOpenGLCudaImage>::New();
			
			textureObject->SetContext((*render_widgets.begin())->GetRenderWindow());
			//texture->SetTextureObject(textureObject);
		}
		EagleLib::cuda::enqueue_callback_async(
			[this, d_mat, stream]()->void
		{
			//boost::recursive_mutex::scoped_lock lock(this->mtx);
			Parameters::UI::UiCallbackService::Instance()->post(
				boost::bind<void>([d_mat, this, stream]()->void
			{
				textureObject->image_buffer.copyFrom(d_mat, *stream, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
				textureObject->map_gpu_mat(d_mat);
				texture->SetTextureObject(textureObject);
			}));
			
		}, *stream);
	}
	
}

std::string vtkImageViewer::PlotName() const
{
	return "vtkImageViewer";
}

REGISTERCLASS(vtkImageViewer);