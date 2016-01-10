#pragma once

#include "plotters/Plotter.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class QVTKWidget;
namespace pcl
{
	namespace visualization
	{
		class PCLVisualizer;
	}
}
namespace EagleLib
{
    class PtCloudDisplay : public QtPlotter
	{
		size_t frameNum;
		std::shared_ptr<Parameters::ITypedParameter<cv::cuda::GpuMat>> gpuParam;
		std::shared_ptr<QVTKWidget> renderWidget;
		std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
		pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud;
	public:
		PtCloudDisplay();
        /*
		virtual QWidget* getPlot(QWidget* parent);
		virtual bool AcceptsParameter(Parameters::Parameter::Ptr param);
		virtual std::string plotName() const;
		virtual QWidget* getSettingsWidget() const;
		
		virtual void doUpdate();
		virtual void setInput(Parameters::Parameter::Ptr param_);
        */
        virtual void AddPlot(QWidget *plot_);
        virtual void SetInput(Parameters::Parameter::Ptr param_ = Parameters::Parameter::Ptr());
        virtual bool AcceptsParameter(Parameters::Parameter::Ptr param);
        virtual void OnParameterUpdate(cv::cuda::Stream* stream);
        virtual std::string PlotName() const;
        virtual PlotterType Type() const;
        virtual QWidget* CreatePlot(QWidget* parent);
        virtual QWidget* GetControlWidget(QWidget* parent);
    };
}
