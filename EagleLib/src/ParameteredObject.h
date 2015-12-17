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

        template<typename T> 
        size_t registerParameter(const std::string& name, T* data, Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, const std::string& toolTip_ = std::string(), bool ownsData = false);
        
        template<typename T> 
        size_t addParameter(const std::string& name, const T& data, Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, const std::string& toolTip_ = std::string(), bool ownsData = false);

        template<typename T> 
        size_t addInputParameter(const std::string& name, const std::string& toolTip_ = std::string(), const boost::function<bool(Parameters::Parameter*)>& qualifier_ = boost::function<bool(Parameters::Parameter*)>());

        template<typename T> 
        bool updateInputQualifier(const std::string& name, const boost::function<bool(Parameters::Parameter::Ptr&)>& qualifier_);
        
        template<typename T> 
        bool updateInputQualifier(int idx, const boost::function<bool(Parameters::Parameter*)>& qualifier_);

        template<typename T> 
        bool updateParameterPtr(const std::string& name, T* data, Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, const std::string& toolTip_ = std::string(), const bool ownsData_ = false, cv::cuda::Stream* stream = nullptr);

        template<typename T> bool updateParameter(const std::string& name, const T& data, Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, const std::string& toolTip_ = std::string(), const bool& ownsData_ = false, cv::cuda::Stream* stream = nullptr);

        template<typename T> bool updateParameter(const std::string& name, const T& data, cv::cuda::Stream* stream);

        template<typename T> bool updateParameter(size_t idx, const T data, const std::string& name = std::string(), const std::string quickHelp = std::string(), Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, cv::cuda::Stream* stream = nullptr);

        template<typename T> bool updateParameter(size_t idx, const T data, cv::cuda::Stream* stream);

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameter(std::string name);

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameterOptional(std::string name);

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameter(int idx);     

        template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameterOptional(int idx);
    protected:
        std::vector<Parameters::Parameter::Ptr> parameters;
    };

}
#include "ParameteredObject_impl.hpp"