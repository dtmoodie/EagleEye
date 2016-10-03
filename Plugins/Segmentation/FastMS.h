#pragma once
#ifdef FASTMS_FOUND
#include "EagleLib/Nodes/Node.h"
#include <MetaObject/Parameters/ParameterMacros.hpp>
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
        MO_DERIVE(FastMumfordShah, Node)
            PARAM(double, lambda, 0.1);
            PARAM(double, alpha, 20.0);
            PARAM(double, temporal, 0.0);
            PARAM(int, interations, 10000);
            PARAM(double, epsilon, 5e-5);
            PARAM(int, stop_k, 10);
            PARAM(bool, adapt_params, false);
            PARAM(bool, weight, false);
            PARAM(bool, overlay_edges, false);
        MO_END;


        FastMumfordShah();
        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer* serializer);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
    };
    }
}
#endif //  FASTMS_FOUND
