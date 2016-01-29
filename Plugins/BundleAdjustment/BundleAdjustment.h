#pragma once
#define  GLOG_NO_ABBREVIATED_SEVERITIES
#include "nodes/Node.h"
#include <iostream>
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <ObjectInterfacePerModule.h>
#include "Manager.h"
#include <ceres/ceres.h>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib
{
	class BundleAdjustment : public Node
	{
	public:
		BundleAdjustment();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};
}
