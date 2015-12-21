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
    /*class PtCloudDisplay : public QtPlotter
	{
		size_t frameNum;
		std::shared_ptr<Parameters::ITypedParameter<cv::cuda::GpuMat>> gpuParam;
		std::shared_ptr<QVTKWidget> renderWidget;
		std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
		pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud;
	public:
		PtCloudDisplay();
		virtual QWidget* getPlot(QWidget* parent);
		virtual bool acceptsWidget(QWidget *widget);
		virtual bool acceptsType(Parameters::Parameter::Ptr param) const;
		virtual std::string plotName() const;
		virtual QWidget* getSettingsWidget() const;
		virtual void addPlot(QWidget *plot_);
		virtual void doUpdate();
		virtual void setInput(Parameters::Parameter::Ptr param_);
    };*/
}
