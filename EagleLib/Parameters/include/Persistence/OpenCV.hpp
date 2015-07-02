#pragma once


// Check if being built with OpenCV

#include <map>
#include <functional>
#include <opencv2/core/persistence.hpp>
#include "../Parameters.hpp"

namespace Parameters
{
	class InterpreterRegistry
	{
		// Mapping from Loki::typeinfo to file writing functors
		//std::map<Loki::TypeInfo, std::function<void(cv::FileStorage&, Parameters::Parameter*)>> registry;
	public:
		//void RegisterFunction(Loki::TypeInfo& type, std::function<void(cv::FileStorage&, Parameter*)> f);
	};

	class cvFileWriter
	{

	};

	template<typename T> void Serializer(cv::FileStorage& fs, Parameters::Parameter* param)
	{
		ITypedParameter<T>* typedParam = dynamic_cast<ITypedParameter<T>*>(param);
		if (typedParam)
		{
			fs << param->GetName();
			fs << *typedParam->Data();
		}
	}

	template<typename T> class cvPersistencePolicy
	{
		cvPersistencePolicy()
		{
			//InterpreterRegistry::RegisterFunction(Loki::TypeInfo(typeid(T)), std::bind(Serializer<T>, _1, _2));
		}
	};
}
