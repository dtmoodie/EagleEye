#include "PCL_bridge.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")
using namespace EagleLib;
IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}



cv::cuda::GpuMat PCL_bridge::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    return img;
}
void PCL_bridge::Init(bool firstInit)
{

}


cv::cuda::GpuMat HuMoments::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	auto d_input = getParameter<cv::cuda::GpuMat>(0);
	if (d_input)
	{
		auto d_mask = getParameter<cv::cuda::GpuMat>(1);

	}
	return img;
}
void HuMoments::Init(bool firstInit)
{
	if (firstInit)
	{
		addInputParameter<cv::cuda::GpuMat>("Device point Cloud");
		addInputParameter<cv::cuda::GpuMat>("Device mask");
		addInputParameter<pcl::PointCloud<pcl::PointXYZ>::Ptr>("Host point cloud");
		addInputParameter<cv::Mat>("Host mask");
	}
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(PCL_bridge, Testing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(HuMoments, PtCloud, Extractor)
