#pragma once
#include "EagleLib/nodes/Node.h"
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
namespace Nodes
{
    class FrameRate: public Node
    {
    public:
        MO_DERIVE(FrameRate, Node);
            MO_PARAM(double, framerate, 30.0);
        MO_END;
    protected:
        bool ProcessImpl();  
    };
    
    class FrameLimiter : public Node
    {
    public:
        
    protected:
        bool ProcessImpl();
    };

    class CreateMat: public Node
    {
    public:
    protected:
        bool ProcessImpl();
        
    };
    class SetMatrixValues: public Node
    {
    public:

    protected:
        bool ProcessImpl();
        
    };
    class Resize : public Node
    {
    protected:
        bool ProcessImpl();
    };
    class Subtract : public Node
    {
    public:
    protected:
        bool ProcessImpl();
    };
}
}
