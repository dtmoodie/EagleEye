#include "slam.hpp"
#include "ObjectInterfacePerModule.h"
SETUP_PROJECT_IMPL

#include <nodes/Node.h>

#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"

#include "utils/utils.hpp"
namespace EagleLib
{
	class DTAM : public Node
	{
		cv::Mat cameraMatrix, R, T;
		std::shared_ptr<CostVolume> cost_volume;
		std::shared_ptr<cv::cuda::DepthmapDenoiseWeightedHuber> dp;
		std::shared_ptr<Optimizer> optimizer;
		size_t image_index;

	public:
		DTAM();
		virtual void Init(bool firstInit)
		{
			if (firstInit)
			{
				image_index = 0;
				updateParameter("Num layers", 32);
				updateParameter("Num images per cost volume", 2);
				updateParameter("Reconstruction scale", 1.0);
			}
		}
		virtual void Serialize(ISimpleSerializer* pSerializer)
		{
			Node::Serialize(pSerializer);
			SERIALIZE(cost_volume);
			SERIALIZE(dp);
			SERIALIZE(optimizer);
			SERIALIZE(image_index);
		}
		virtual cv::cuda::GpuMat doProcessdoProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
		{
			cv::Mat h_img;
			img.download(h_img, stream);
			if (cost_volume == nullptr)
			{
				cost_volume.reset(new CostVolume());
			}
			CV_Assert(cost_volume);
			if (cost_volume->count < *getParameter<int>(1)->Data())
			{
				cost_volume->updateCost(h_img, R, T);
				return img;
			}

			if (optimizer == nullptr)
			{
				optimizer.reset(new Optimizer(*cost_volume));
			}
			CV_Assert(optimizer);
			if (depth_optimizer == nullptr)
			{
				depth_optimizer = cv::cuda::createDepthmapDenoiseWeightedHuber()
			}
			return img;
		}

	};
}
using namespace EagleLib;
NODE_DEFAULT_CONSTRUCTOR_IMPL(DTAM);