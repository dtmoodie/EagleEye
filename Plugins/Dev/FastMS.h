#include "nodes/Node.h"
#include "solver.h"
namespace EagleLib
{
	class SegmentFastMumfordShah : public Node
	{
		cv::cuda::HostMem h_img;
		Solver solver;
	public:
		SegmentFastMumfordShah();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
}