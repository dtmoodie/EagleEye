#include "visualizer.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vtkRenderWindow.h>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda.hpp>
#include <QVTKWidget.h>

#include <UI/InterThread.hpp>

using namespace EagleLib;
/*
PtCloudDisplay::PtCloudDisplay()
{
	frameNum = 0;
}
QWidget* PtCloudDisplay::getPlot(QWidget* parent)
{
	if (renderWidget == nullptr)
	{
		renderWidget.reset(new QVTKWidget(parent));
	}		
	if (viewer == nullptr)
	{
		viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
		renderWidget->SetRenderWindow(viewer->getRenderWindow());
		viewer->addCoordinateSystem();
		renderWidget->update();
	}

	return renderWidget.get();

}
bool PtCloudDisplay::acceptsWidget(QWidget *widget)
{
	return false;
}
bool PtCloudDisplay::acceptsType(Parameters::Parameter::Ptr param) const
{
	auto gpuParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param);
	if (gpuParam)
	{
		cv::cuda::GpuMat* gpuMat = gpuParam->Data();
		if (gpuMat)
		{
			if (gpuMat->depth() == CV_32F)
			{
				if (gpuMat->channels() == 4)
				{
					// XYZ is packed into channels, not tensor format
					return true;
				}
				else
				{
					if (gpuMat->cols == 4)
					{
						// XYZ is packed into tensor format
						return true;
					}
				}
			}
		}
	}
	return false;
}
std::string PtCloudDisplay::plotName() const
{
	return "PtCloudDisplay";
}
QWidget* PtCloudDisplay::getSettingsWidget() const
{
	return nullptr;
}
void PtCloudDisplay::addPlot(QWidget *plot_)
{

}


void PtCloudDisplay::doUpdate()
{
	if (gpuParam)
	{
		cv::cuda::GpuMat* gpuMat = gpuParam->Data();
		if (gpuMat)
		{
			size_t numPoints = 0;
			if (gpuMat->channels() == 1 && gpuMat->cols == 4)
			{
				numPoints = gpuMat->rows;
			}
			else
			{
				if (gpuMat->channels() == 4) 
				{
					numPoints = gpuMat->rows * gpuMat->cols;
				}
			}
			if (numPoints == 0)
				return;
			if (inputCloud == nullptr)
			{
				inputCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
			}
			inputCloud->resize(numPoints);
			cv::Mat mappedMat(gpuMat->rows, gpuMat->cols, CV_MAKETYPE(CV_32F, gpuMat->channels()), &(*inputCloud)[0]);
			gpuMat->download(mappedMat);
			inputCloud->height = gpuMat->rows;
			inputCloud->width = gpuMat->cols;
			inputCloud->is_dense = gpuMat->cols != 4;
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(inputCloud, 0, 255, 0);
			if (frameNum == 0)
			{
				viewer->addPointCloud<pcl::PointXYZ>(inputCloud, single_color, "cloud");
			}
			else
			{
				viewer->updatePointCloud<pcl::PointXYZ>(inputCloud, single_color, "cloud");
			}
			++frameNum;
			Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&QVTKWidget::update, renderWidget.get()));
		}
	}
}
void PtCloudDisplay::setInput(Parameters::Parameter::Ptr param_)
{
	if (param_)
	{
		gpuParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param_);
		
		param_->RegisterNotifier(boost::bind(&Parameters::UI::UiCallbackService::post, Parameters::UI::UiCallbackService::Instance(), boost::function<void(void)>(boost::bind(&PtCloudDisplay::doUpdate, this))));
		doUpdate();
	}
	else
	{
		gpuParam.reset();
	}
	
}*/

REGISTERCLASS(PtCloudDisplay)
void PtCloudDisplay::SetInput(Parameters::Parameter::Ptr param_)
{
    gpuParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param_);
    param_->RegisterNotifier(boost::bind(&PtCloudDisplay::OnParameterUpdate, this, _1));
}
bool PtCloudDisplay::AcceptsParameter(Parameters::Parameter::Ptr param)
{
    auto typeinfo = param->GetTypeInfo();
    if (typeinfo == Loki::TypeInfo(typeid(cv::cuda::GpuMat)))
    {
        auto temp_gpu_param = std::dynamic_pointer_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>>(param);
        auto size = temp_gpu_param->Data()->size();
        auto type = temp_gpu_param->Data()->type();
        if (size.width == 3 || size.width == 4 && type == CV_32F)
        {
            return true;
        }
        if (type == CV_32FC3)
        {
            return true;
        }
    }
    return false;
    
}
void PtCloudDisplay::OnParameterUpdate(cv::cuda::Stream* stream)
{
    if (gpuParam)
    {
        cv::cuda::GpuMat* gpuMat = gpuParam->Data();
        if (gpuMat)
        {
            size_t numPoints = 0;
            if (gpuMat->channels() == 1 && gpuMat->cols == 4)
            {
                numPoints = gpuMat->rows;
            }
            else
            {
                if (gpuMat->channels() == 4)
                {
                    numPoints = gpuMat->rows * gpuMat->cols;
                }
            }
            if (numPoints == 0)
                return;
            if (inputCloud == nullptr)
            {
                inputCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
            }
            inputCloud->resize(numPoints);
            cv::Mat mappedMat(gpuMat->rows, gpuMat->cols, CV_MAKETYPE(CV_32F, gpuMat->channels()), &(*inputCloud)[0]);
            gpuMat->download(mappedMat);
            inputCloud->height = gpuMat->rows;
            inputCloud->width = gpuMat->cols;
            inputCloud->is_dense = gpuMat->cols != 4;
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(inputCloud, 0, 255, 0);
            if (frameNum == 0)
            {
                viewer->addPointCloud<pcl::PointXYZ>(inputCloud, single_color, "cloud");
            }
            else
            {
                viewer->updatePointCloud<pcl::PointXYZ>(inputCloud, single_color, "cloud");
            }
            ++frameNum;
            Parameters::UI::UiCallbackService::Instance()->post(boost::bind(&QVTKWidget::update, renderWidget.get()));
        }
    }
}

std::string PtCloudDisplay::PlotName() const
{
    return "PtCloudDisplay";
}

QWidget* PtCloudDisplay::CreatePlot(QWidget* parent)
{
    if (renderWidget == nullptr)
    {
        renderWidget.reset(new QVTKWidget(parent));
    }
    if (viewer == nullptr)
    {
        viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
        renderWidget->SetRenderWindow(viewer->getRenderWindow());
        viewer->addCoordinateSystem();
        renderWidget->update();
    }

    return renderWidget.get();
}

QWidget* PtCloudDisplay::GetControlWidget(QWidget* parent)
{
    return nullptr;
}