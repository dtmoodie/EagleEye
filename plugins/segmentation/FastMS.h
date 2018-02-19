#pragma once
#ifdef FASTMS_FOUND
#include "Aquila/nodes/Node.hpp"
#include <MetaObject/params/ParameterMacros.hpp>
#include "libfastms/solver/solver.h"
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("-lfastmsd")
#else
RUNTIME_COMPILER_LINKLIBRARY("-lfastms")
#endif
namespace aq
{
    namespace nodes
    {

    class FastMumfordShah : public Node
    {
    public:
        MO_DERIVE(FastMumfordShah, Node)
            INPUT(SyncedMemory, input, nullptr)
            PARAM(double, lambda, 0.1)
            PARAM(double, alpha, 20.0)
            PARAM(double, temporal, 0.0)
            PARAM(int, iterations, 10000)
            PARAM(double, epsilon, 5e-5)
            PARAM(int, stop_k, 10)
            PARAM(bool, adapt_params, false)
            PARAM(bool, weight, false)
            PARAM(bool, overlay_edges, false)
            PROPERTY(boost::shared_ptr<Solver> , solver, boost::shared_ptr<Solver> ())
            OUTPUT(SyncedMemory, segmented, SyncedMemory())
        MO_END;
    protected:
        bool processImpl();
    };
    }
}
#endif //  FASTMS_FOUND
