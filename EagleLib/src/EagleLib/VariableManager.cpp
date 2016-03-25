#include "VariableManager.h"
#include "parameters/Parameter.hpp"
#include "parameters/InputParameter.hpp"
#include <signals/logging.hpp>
using namespace EagleLib;
void VariableManager::AddParameter(std::shared_ptr<Parameters::Parameter> param)
{
    _parameters[param->GetTreeName()] = param;
}
void VariableManager::RemoveParameter(std::shared_ptr<Parameters::Parameter> param)
{
    for(auto itr = _parameters.begin(); itr != _parameters.end(); ++itr)
    {
        if(itr->second == param)
        {
            _parameters.erase(itr);
            return;
        }
    }
}
std::vector<std::shared_ptr<Parameters::Parameter>> VariableManager::GetOutputParameters(Loki::TypeInfo type)
{
    std::vector<std::shared_ptr<Parameters::Parameter>> valid_outputs;
    for(auto itr = _parameters.begin(); itr != _parameters.end(); ++itr)
    {
        if(itr->second->GetTypeInfo() == type && itr->second->type & Parameters::Parameter::Output)
        {
            valid_outputs.push_back(itr->second);
        }
    }
    return valid_outputs;
}
std::shared_ptr<Parameters::Parameter> VariableManager::GetOutputParameter(std::string name)
{
    auto itr = _parameters.find(name);
    if(itr != _parameters.end())
    {
        return itr->second;
    }
    LOG(warning) << "Unable to find parameter named " << name;
    return std::shared_ptr<Parameters::Parameter>();
}
void VariableManager::LinkParameters(std::shared_ptr<Parameters::Parameter> output, std::shared_ptr<Parameters::Parameter> input)
{
	if(auto input_param = std::dynamic_pointer_cast<Parameters::InputParameter>(input))
		input_param->SetInput(output);
}