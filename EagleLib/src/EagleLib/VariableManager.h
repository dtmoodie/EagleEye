#pragma once

#include "IVariableManager.h"
#include <map>

namespace EagleLib
{
    class VariableManager:public IVariableManager
    {
        std::map<std::string, std::shared_ptr<Parameters::Parameter>> _parameters;        
    public:
        virtual void AddParameter(std::shared_ptr<Parameters::Parameter> param);
		virtual void RemoveParameter(std::shared_ptr<Parameters::Parameter> param);
		virtual std::vector<std::shared_ptr<Parameters::Parameter>> GetOutputParameters(Loki::TypeInfo type);
        virtual std::shared_ptr<Parameters::Parameter> GetOutputParameter(std::string name);
    };
}