#pragma once
#ifdef FASTMS_FOUND
#include "nodes/Node.h"
#include "libfastms/solver/solver.h"
namespace EagleLib
{

	class SegmentFastMumfordShah : public Node
	{
		cv::cuda::HostMem h_img;
		boost::shared_ptr<Solver> solver;
	public:
		SegmentFastMumfordShah();
		virtual void Init(bool firstInit);
		virtual void Serialize(ISimpleSerializer* serializer);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
}
#endif //  FASTMS_FOUND
