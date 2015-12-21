#pragma once
#include "EagleLib/Defs.hpp"
#include "IObject.h"
#include "Parameter.hpp"
#include "InputParameter.hpp"
#include "TypedParameter.hpp"
#include "TypedInputParameter.hpp"
#include <opencv2/core/persistence.hpp>
#include "type.h"
namespace EagleLib
{
    struct ParameteredObjectImpl; // Private implementation stuffs
    class EAGLE_EXPORTS ParameteredObject: public IObject
    {
    public:
        ParameteredObject();
        ~ParameteredObject();
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(const cv::FileNode& configNode);
        virtual void onUpdate(cv::cuda::Stream* stream);

        virtual Parameters::Parameter* addParameter(Parameters::Parameter::Ptr param);

        // Thows exception on unable to get parameter
        Parameters::Parameter::Ptr getParameter(int idx);
        Parameters::Parameter::Ptr getParameter(const std::string& name);
        
        // Returns nullptr on unable to get parameter
        Parameters::Parameter::Ptr getParameterOptional(int idx);
        Parameters::Parameter::Ptr getParameterOptional(const std::string& name);

        void RegisterParameterCallback(int idx, const boost::function<void(cv::cuda::Stream*)>& callback);
        void RegisterParameterCallback(const std::string& name, const boost::function<void(cv::cuda::Stream*)>& callback);
        void RegisterParameterCallback(Parameters::Parameter* param, const boost::function<void(cv::cuda::Stream*)>& callback);
        
        

        template<typename T> 
        Parameters::Parameter* registerParameter(const std::string& name, T* data);
        
        template<typename T> 
        Parameters::Parameter* addParameter(const std::string& name, const T& data);

        template<typename T> 
        Parameters::TypedInputParameter<T>* addInputParameter(const std::string& name);

        template<typename T> 
        bool updateInputQualifier(const std::string& name, const boost::function<bool(Parameters::Parameter*)>& qualifier);
        
        template<typename T> 
        bool updateInputQualifier(int idx, const boost::function<bool(Parameters::Parameter*)>& qualifier);

        template<typename T> 
        Parameters::Parameter* updateParameterPtr(const std::string& name, T* data, cv::cuda::Stream* stream = nullptr);

        template<typename T> 
        Parameters::Parameter* updateParameter(const std::string& name, const T& data, cv::cuda::Stream* stream = nullptr);

        template<typename T> 
        Parameters::Parameter* updateParameter(size_t idx, const T data, cv::cuda::Stream* stream = nullptr);

        template<typename T> 
        typename Parameters::ITypedParameter<T>::Ptr getParameter(std::string name);

        template<typename T>
        typename Parameters::ITypedParameter<T>::Ptr getParameter(int idx);

        template<typename T> 
        typename Parameters::ITypedParameter<T>::Ptr getParameterOptional(std::string name);

        template<typename T> 
        typename Parameters::ITypedParameter<T>::Ptr getParameterOptional(int idx);

        

        std::vector<Parameters::Parameter::Ptr> parameters;
    protected:
        
        struct Impl;
        std::shared_ptr<ParameteredObjectImpl> _impl;
    };







    // ****************************** Implementation ****************************************
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
    Parameters::TypedInputParameter<T>* ParameteredObject::addInputParameter(const std::string& name)
    {
        auto input_param = new Parameters::TypedInputParameter<T>(name);
        addParameter(typename Parameters::TypedInputParameter<T>::Ptr(input_param));
        return input_param;
    }

    template<typename T>
    bool ParameteredObject::updateInputQualifier(const std::string& name, const boost::function<bool(Parameters::Parameter*)>& qualifier_)
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
    bool ParameteredObject::updateInputQualifier(int idx, const boost::function<bool(Parameters::Parameter*)>& qualifier)
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
        if (idx > parameters.size() || idx < 0)
            return nullptr;
        auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<T>>(parameters[idx]);
        if (param == NULL)
            return nullptr;
        param->UpdateData(data, stream);
        onUpdate(stream);
        return param.get();
    }

    template<typename T>
    typename Parameters::ITypedParameter<T>::Ptr ParameteredObject::getParameter(std::string name)
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
    typename Parameters::ITypedParameter<T>::Ptr ParameteredObject::getParameter(int idx)
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
    typename Parameters::ITypedParameter<T>::Ptr ParameteredObject::getParameterOptional(std::string name)
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
    typename Parameters::ITypedParameter<T>::Ptr ParameteredObject::getParameterOptional(int idx)
    {
        auto param = getParameterOptional(idx);
        if (param == nullptr)
            return typename Parameters::ITypedParameter<T>::Ptr(); // Return a nullptr

        return std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
    }
}
