#include <EagleLib/Nodes/Node.h>
#include "EagleLib/Detail/Export.hpp"
#include <EagleLib/Project_defs.hpp>
SETUP_PROJECT_DEF

namespace EagleLib
{
    namespace Nodes
    {
    class nvcc_test: public Node
    {
    public:
        MO_DERIVE(nvcc_test, Node)
            INPUT(SyncedMemory, input, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    }
}
