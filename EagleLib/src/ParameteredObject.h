#pragma once

#include "IObject.h"
#include "Parameter.hpp"
namespace EagleLib
{
    class ParameteredObject: public IObject
    {
    public:
        virtual void Serialize(ISimpleSerializer* pSerializer);

        size_t addParameter(Parameters::Parameter::Ptr param);
        virtual Parameters::Parameter::Ptr getParameter(int idx);
        virtual Parameters::Parameter::Ptr getParameter(const std::string& name);
        void RegisterParameterCallback(int idx, boost::function<void(cv::cuda::Stream*)> callback);
        void RegisterParameterCallback(const std::string& name, boost::function<void(cv::cuda::Stream*)> callback);
        void RegisterSignalConnection(boost::signals2::connection& connection);

        template<typename T> size_t registerParameter(
            const std::string& name,
            T* data,
            Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
            const std::string& toolTip_ = std::string(),
            bool ownsData = false)
        {
            return addParameter(typename Parameters::TypedParameterPtr<T>::Ptr(new Parameters::TypedParameterPtr<T>(name, data, type_, toolTip_)));
        }
        
        template<typename T> size_t addParameter(const std::string& name,
            const T& data,
            Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
            const std::string& toolTip_ = std::string(), 
            bool ownsData = false)
        {
            return addParameter(typename Parameters::TypedParameter<T>::Ptr(new Parameters::TypedParameter<T>(name, data, type_, toolTip_)));
        }

        template<typename T> size_t
            addInputParameter(const std::string& name, const std::string& toolTip_ = std::string(), const boost::function<bool(Parameters::Parameter*)>& qualifier_ = boost::function<bool(Parameters::Parameter*)>())
        {
            return addParameter(typename Parameters::TypedInputParameter<T>::Ptr(new Parameters::TypedInputParameter<T>(name, toolTip_, qualifier_)));
        }

        template<typename T> bool
            updateInputQualifier(const std::string& name, const boost::function<bool(Parameters::Parameter::Ptr&)>& qualifier_)
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
        
        template<typename T> bool
            updateInputQualifier(int idx, const boost::function<bool(Parameters::Parameter*)>& qualifier_)
        {
            auto param = getParameter<T>(idx);
            if(param && param->type & Parameters::Parameter::Input)
            {
                Parameters::InputParameter* inputParam = dynamic_cast<Parameters::InputParameter*>(param.get());
                if(inputParam)
                {
                    inputParam->SetQualifier(qualifier_);
                    return true;
                }
            }
            return false;
        }

        template<typename T> bool updateParameterPtr(
            const std::string& name, 
            T* data, 
            Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
            const std::string& toolTip_ = std::string(), 
            const bool ownsData_ = false, cv::cuda::Stream* stream = nullptr)
        {
            typename Parameters::ITypedParameter<T>::Ptr param;
            try
            {
                param = getParameter<T>(name);
            }
            catch (cv::Exception &e)
            {
                e.what();
                BOOST_LOG_TRIVIAL(debug) << name << " doesn't exist, adding";
                return registerParameter<T>(name, data, type_, toolTip_, ownsData_);
            }
            param->UpdateData(data);
            onUpdate(stream);
            return true;

        }

        template<typename T> bool updateParameter(const std::string& name,
            const T& data,
            Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
            const std::string& toolTip_ = std::string(),
            const bool& ownsData_ = false, cv::cuda::Stream* stream = nullptr)
        {
            typename Parameters::ITypedParameter<T>::Ptr param;
            param = getParameterOptional<T>(name);
            if (param == nullptr)
            {
                BOOST_LOG_TRIVIAL(debug) << "Parameter named \"" << name << "\" with type " << Loki::TypeInfo(typeid(T)).name() << " doesn't exist, adding";
                return addParameter<T>(name, data, type_, toolTip_, ownsData_);
            }

            if (type_ != Parameters::Parameter::None)
                param->type = type_;
            if (toolTip_.size() > 0)
                param->SetTooltip(toolTip_);

            param->UpdateData(data, stream);
            onUpdate(stream);
            return true;
        }
        template<typename T> bool updateParameter(const std::string& name,
            const T& data,
            cv::cuda::Stream* stream)
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
            return true;
        }

        template<typename T> bool updateParameter(size_t idx,
            const T data,
            const std::string& name = std::string(),
            const std::string quickHelp = std::string(),
            Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, cv::cuda::Stream* stream = nullptr)
        {
            if (idx > parameters.size() || idx < 0)
                return false;
            auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<T>>(parameters[idx]);
            if (param == NULL)
                return false;

            if (name.size() > 0)
                param->SetName( name );
            if (type_ != Parameters::Parameter::None)
                param->type = type_;
            if (quickHelp.size() > 0)
                param->SetTooltip(quickHelp);
            param->UpdateData(data,stream);
            onUpdate(stream);
            return true;
        }

        template<typename T> bool updateParameter(size_t idx,
            const T data,
            cv::cuda::Stream* stream)
        {
            if (idx > parameters.size() || idx < 0)
                return false;
            auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<T>>(parameters[idx]);
            if (param == NULL)
                return false;
            param->UpdateData(data, stream);
            onUpdate(stream);
            return true;
        }


        template<typename T> typename Parameters::ITypedParameter<T>::Ptr 	getParameterOptional(std::string name)
        {
            auto param =  getParameter(name);
            if (param == nullptr)
            {
                return typename Parameters::ITypedParameter<T>::Ptr();
            }
            auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
            if(typedParam == nullptr)
                throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                    TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + param->GetTypeInfo().name(), __FUNCTION__, __FILE__, __LINE__);

            return typedParam;
        }

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr 	getParameter(std::string name)
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

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameter(int idx)
        {
            auto param = getParameter(idx);
            if(param == nullptr)
                throw cv::Exception(0, "Failed to get parameter by index " + boost::lexical_cast<std::string>(idx), __FUNCTION__, __FILE__, __LINE__);

            auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
            if(typedParam == nullptr)
                throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                    TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + param->GetTypeInfo().name(), __FUNCTION__, __FILE__, __LINE__);
            return typedParam;
        }

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameterOptional(int idx)
        {
            auto param = getParameter(idx);
            if (param == nullptr)
                return typename Parameters::ITypedParameter<T>::Ptr(); // Return a nullptr

            return std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
        }
    protected:
        std::vector<Parameters::Parameter::Ptr> parameters;
    };

}