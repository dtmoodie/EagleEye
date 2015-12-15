#pragma once

namespace EagleLib
{
    template<typename T> 
    size_t ParameteredObject::registerParameter(const std::string& name, T* data, Parameters::Parameter::ParameterType type_, const std::string& toolTip_, bool ownsData)
    {
        return addParameter(typename Parameters::TypedParameterPtr<T>::Ptr(new Parameters::TypedParameterPtr<T>(name, data, type_, toolTip_)));
    }

    template<typename T> 
    size_t ParameteredObject::addParameter(const std::string& name,
        const T& data,
        Parameters::Parameter::ParameterType type_,
        const std::string& toolTip_, 
        bool ownsData)
    {
        return addParameter(typename Parameters::TypedParameter<T>::Ptr(new Parameters::TypedParameter<T>(name, data, type_, toolTip_)));
    }
}