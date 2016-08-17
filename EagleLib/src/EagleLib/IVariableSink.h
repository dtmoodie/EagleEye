#pragma once

namespace mo
{
    class IVariableManager;
}
namespace EagleLib
{
    class IVariableSink
    {
    public:
        virtual void SerializeVariables(unsigned long long frame_number, mo::IVariableManager* manager) = 0;
    };
}
