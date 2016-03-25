#pragma once

#include <parameters/LokiTypeInfo.h>
#include <vector>
#include <memory>
#include "EagleLib/Defs.hpp"
namespace Parameters
{
	class Parameter;
	template<typename T> class ITypedParameter;
	template<typename T> class TypedInputParameter;
}
namespace EagleLib
{
	class EAGLE_EXPORTS IVariableManager
	{
	public:
		virtual void AddParameter(std::shared_ptr<Parameters::Parameter> param) = 0;

		virtual void RemoveParameter(std::shared_ptr<Parameters::Parameter> param) = 0;

		virtual std::vector<std::shared_ptr<Parameters::Parameter>> GetOutputParameters(Loki::TypeInfo type) = 0;

		template<typename T> std::vector<std::shared_ptr<Parameters::Parameter>> GetOutputParameters();

        virtual std::shared_ptr<Parameters::Parameter> GetOutputParameter(std::string name) = 0;

		// Links an output parameter to an input parameter with either a direct link or a buffered link.
		// Buffered links allow parameters to desync frame number between the producer and the consumer
		virtual void LinkParameters(std::shared_ptr<Parameters::Parameter> output, std::shared_ptr<Parameters::Parameter> input) = 0;
	};

	template<typename T> std::vector<std::shared_ptr<Parameters::Parameter>> IVariableManager::GetOutputParameters()
	{
		return GetOutputParameters(Loki::TypeInfo(typeid(T)));
	}
}