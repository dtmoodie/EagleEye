#pragma once
#ifdef FASTMS_FOUND
#include "EagleLib/nodes/Node.h"
#include "libfastms/solver/solver.h"
#include "RuntimeLinkLibrary.h"
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("fastmsd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("fastms.lib")
#endif
namespace EagleLib
{
    namespace Nodes
    {
    
	class FastMumfordShah : public Node
	{
		cv::cuda::HostMem h_img;
		boost::shared_ptr<Solver> solver;
	public:
		FastMumfordShah();
		virtual void NodeInit(bool firstInit);
		virtual void Serialize(ISimpleSerializer* serializer);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
    }
}
#endif //  FASTMS_FOUND
