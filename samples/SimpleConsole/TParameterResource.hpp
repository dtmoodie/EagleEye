#ifdef HAVE_WT
#pragma once
#include "IParameterResource.hpp"
#include <EagleLib/SyncedMemory.h>
#include <MetaObject/Parameters/TypedInputParameter.hpp>
namespace vclick
{
    template<class T> class TParameterResource : public IParameterResource
    {
    public:
        TParameterResource(Wt::WApplication* app, mo::IParameter* param, const std::string& name = "default"):
            IParameterResource(app)
        {
            data = nullptr;
            data_param.SetUserDataPtr(&data);
            data_param.SetName(name);
            data_param.SetInput(param);
            data_param.SetMtx(&param->mtx());
            this->setParam(&data_param);
        }
    private:
        const T* data;
        mo::TypedInputParameterPtr<T> data_param;
    };
    template<class T> class TParameterResourceRaw: public IParameterResource
    {
        
    };  

    template<> class TParameterResourceRaw<cv::Mat>: public IParameterResource
    {
    public:
        TParameterResourceRaw(Wt::WApplication* app, mo::IParameter* param, const std::string& name = "default") :
            IParameterResource(app)
        {
            data = nullptr;
            data_param.SetUserDataPtr(&data);
            data_param.SetName(name);
            data_param.SetInput(param);
            data_param.SetMtx(&param->mtx());
            this->setParam(&data_param);
        }
        void handleParamUpdate(mo::Context* ctx, mo::IParameter* param);
    private:
        const cv::Mat* data;
        mo::TypedInputParameterPtr<cv::Mat> data_param;
    };

    template<> class TParameterResource<EagleLib::SyncedMemory> : public IParameterResource
    {
    public:
        TParameterResource(Wt::WApplication* app, mo::IParameter* param,
            const std::string& name = "default") :
            IParameterResource(app)
        {
            data = nullptr;
            data_param.SetUserDataPtr(&data);
            data_param.SetName(name);
            data_param.SetInput(param);
            data_param.SetMtx(&param->mtx());
            this->setParam(&data_param);
        }
        void handleParamUpdate(mo::Context* ctx, mo::IParameter* param);

    private:
        const EagleLib::SyncedMemory* data;
        mo::TypedInputParameterPtr<EagleLib::SyncedMemory> data_param;
    };
}
#endif