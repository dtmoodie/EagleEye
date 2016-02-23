#pragma once
#include "ParameteredObject.h"
#include "parameters/Parameter.hpp"
#include "parameters/InputParameter.hpp"
#include "parameters/TypedParameter.hpp"
#include "parameters/TypedInputParameter.hpp"
#include <boost/lexical_cast.hpp>
#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("libParameterd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("libParameter.lib")
#endif
#else
#endif
namespace EagleLib
{
	template<typename T>
	Parameters::Parameter* ParameteredObject::registerParameter(const std::string& name, T* data)
	{
		return addParameter(typename Parameters::TypedParameterPtr<T>::Ptr(new Parameters::TypedParameterPtr<T>(name, data)));
	}

	template<typename T>
	Parameters::Parameter* ParameteredObject::addParameter(const std::string& name, const T& data)
	{
		return addParameter(typename Parameters::TypedParameter<T>::Ptr(new Parameters::TypedParameter<T>(name, data)));
	}
	template<typename T>
	Parameters::Parameter* ParameteredObject::addIfNotExist(const std::string& name, const T& data)
	{
		if (!exists(name))
			return addParameter(typename Parameters::TypedParameter<T>::Ptr(new Parameters::TypedParameter<T>(name, data)));
		return nullptr;
	}

	template<typename T>
	Parameters::TypedInputParameter<T>* ParameteredObject::addInputParameter(const std::string& name)
	{
		auto input_param = new Parameters::TypedInputParameter<T>(name);
		addParameter(typename Parameters::TypedInputParameter<T>::Ptr(input_param));
		return input_param;
	}

	template<typename T>
	bool ParameteredObject::updateInputQualifier(const std::string& name, const std::function<bool(Parameters::Parameter*)>& qualifier_)
	{
		auto param = getParameter(name);
		if (param && param->type & Parameters::Parameter::Input)
		{
			Parameters::InputParameter* inputParam = dynamic_cast<Parameters::InputParameter*>(param.get());
			if (inputParam)
			{
				inputParam->SetQualifier(qualifier_);
				return true;
			}
		}
		return false;
	}

	template<typename T>
	bool ParameteredObject::updateInputQualifier(int idx, const std::function<bool(Parameters::Parameter*)>& qualifier)
	{
		auto param = getParameter<T>(idx);
		if (param && param->type & Parameters::Parameter::Input)
		{
			Parameters::InputParameter* inputParam = dynamic_cast<Parameters::InputParameter*>(param.get());
			if (inputParam)
			{
				inputParam->SetQualifier(qualifier);
				return true;
			}
		}
		return false;
	}

	template<typename T>
	Parameters::Parameter* ParameteredObject::updateParameterPtr(const std::string& name, T* data, cv::cuda::Stream* stream)
	{
		typename Parameters::ITypedParameter<T>::Ptr param;
		param = getParameterOptional<T>(name);
		if (param == nullptr)
		{
			BOOST_LOG_TRIVIAL(debug) << name << " doesn't exist, adding";
			return registerParameter<T>(name, data);
		}
		if (auto non_ref_param = std::dynamic_pointer_cast<Parameters::TypedParameter<T>>(param))
		{
			// Parameter exists but is of the wrong type, due to being loaded from a file as a generic typed parameter
			*data = *non_ref_param->Data();
			// Find the incorrectly typed parameter
			for (int i = 0; i < _parameters.size(); ++i)
			{
				if (_parameters[i]->GetName() == name)
				{
					_parameters[i] = typename Parameters::TypedParameterPtr<T>::Ptr(new Parameters::TypedParameterPtr<T>(name, data));
				}
			}
		}
		param->UpdateData(data);
		onUpdate(stream);
		return param.get();
	}

	template<typename T>
	Parameters::Parameter* ParameteredObject::updateParameter(const std::string& name, const T& data, cv::cuda::Stream* stream)
	{
		typename Parameters::ITypedParameter<T>::Ptr param;
		param = getParameterOptional<T>(name);
		if (param == nullptr)
		{
			BOOST_LOG_TRIVIAL(debug) << "Parameter named \"" << name << "\" with type " << Loki::TypeInfo(typeid(T)).name() << " doesn't exist, adding";
			return addParameter<T>(name, data);
		}
		param->UpdateData(data, stream);
		onUpdate(stream);
		return param.get();
	}

	template<typename T>
	Parameters::Parameter* ParameteredObject::updateParameter(size_t idx, const T data, cv::cuda::Stream* stream)
	{
		if (idx > _parameters.size() || idx < 0)
			return nullptr;
		auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<T>>(_parameters[idx]);
		if (param == NULL)
			return nullptr;
		param->UpdateData(data, stream);
		onUpdate(stream);
		return param.get();
	}

	template<typename T>
	typename std::shared_ptr<Parameters::ITypedParameter<T>> ParameteredObject::getParameter(std::string name)
	{
		auto param = getParameter(name);
		if (param == nullptr)
		{
			throw cv::Exception(0, "Failed to get parameter by name " + name, __FUNCTION__, __FILE__, __LINE__);
			return typename Parameters::ITypedParameter<T>::Ptr();
		}
		auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
		if (typedParam == nullptr)
			throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
			TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + param->GetTypeInfo().name(), __FUNCTION__, __FILE__, __LINE__);

		return typedParam;
	}

	template<typename T>
	typename std::shared_ptr<Parameters::ITypedParameter<T>> ParameteredObject::getParameter(int idx)
	{
		auto param = getParameter(idx);
		if (param == nullptr)
			throw cv::Exception(0, "Failed to get parameter by index " + boost::lexical_cast<std::string>(idx), __FUNCTION__, __FILE__, __LINE__);

		auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
		if (typedParam == nullptr)
			throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
			TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + param->GetTypeInfo().name(), __FUNCTION__, __FILE__, __LINE__);
		return typedParam;
	}


	template<typename T>
	typename std::shared_ptr<Parameters::ITypedParameter<T>> ParameteredObject::getParameterOptional(std::string name)
	{
		auto param = getParameterOptional(name);
		if (param == nullptr)
		{
			return typename Parameters::ITypedParameter<T>::Ptr();
		}
		auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
		if (typedParam == nullptr)
			BOOST_LOG_TRIVIAL(debug) << "Failed to cast parameter to the appropriate type, requested type: " <<
			TypeInfo::demangle(typeid(T).name()) << " parameter actual type: " << param->GetTypeInfo().name();

		return typedParam;
	}

	template<typename T>
	typename std::shared_ptr<Parameters::ITypedParameter<T>> ParameteredObject::getParameterOptional(int idx)
	{
		auto param = getParameterOptional(idx);
		if (param == nullptr)
			return typename Parameters::ITypedParameter<T>::Ptr(); // Return a nullptr

		return std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
	}
}