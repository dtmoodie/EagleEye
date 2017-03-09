#pragma once

#include <Aquila/Nodes/Node.h>
//#include <Aquila/Nodes/VideoProc/Tracking.hpp>
#include <boost/circular_buffer.hpp>

#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <MetaObject/Parameters/TypedInputParameter.hpp>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace Nodes
    {
    
    class KeyFrameTracker: public Node
    {
    public:
        MO_DERIVE(KeyFrameTracker, Node)
            INPUT(SyncedMemory, input_image, nullptr);
            INPUT(SyncedMemory, input_mask, nullptr);
            PARAM(int, frames_to_track, 5);
            PARAM(double, upper_quality, 0.7);
            PARAM(double, lower_quality, 0.4);
            PARAM(int, min_keypoints, 200);
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class CMT: public Node
    {
    public:
        MO_DERIVE(CMT, Node);
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class TLD:public Node
    {
    public:
        MO_DERIVE(TLD, Node);
        MO_END;
    protected:
        bool ProcessImpl();
        
    };
    }
}
