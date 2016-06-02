#pragma once

namespace Parameters
{
    class IVariableManager;
}
namespace EagleLib
{
    class IVariableSink
{
public:
    virtual void SerializeVariables(unsigned long long frame_number, Parameters::IVariableManager* manager) = 0;
};
}
