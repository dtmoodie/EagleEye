#ifdef HAVE_WT
#pragma once
#include "IParameterResource.hpp"
#include <Aquila/types/SyncedMemory.hpp>
#include <MetaObject/params/TypedInputParam.hpp>
namespace vclick
{
    template<class T> class TParameterResource : public IParameterResource
    {
    public:
        TParameterResource(Wt::WApplication* app, mo::IParam* param, const std::string& name = "default"):
            IParameterResource(app)
        {
            data = nullptr;
            data_param.SetUserDataPtr(&data);
            data_param.setName(name);
            data_param.setInput(param);
            data_param.SetMtx(&param->mtx());
            this->setParam(&data_param);
        }
    private:
        const T* data;
        mo::TypedInputParamPtr<T> data_param;
    };
    template<class T> class TParameterResourceRaw: public IParameterResource
    {
        
    };  

    template<> class TParameterResourceRaw<cv::Mat>: public IParameterResource
    {
    public:
        TParameterResourceRaw(Wt::WApplication* app, mo::IParam* param, const std::string& name = "default") :
            IParameterResource(app)
        {
            data = nullptr;
            data_param.SetUserDataPtr(&data);
            data_param.setName(name);
            data_param.setInput(param);
            data_param.SetMtx(&param->mtx());
            this->setParam(&data_param);
        }
        void handleParamUpdate(mo::Context* ctx, mo::IParam* param);
    private:
        const cv::Mat* data;
        mo::TypedInputParamPtr<cv::Mat> data_param;
    };

    template<> class TParameterResource<aq::SyncedMemory> : public IParameterResource
    {
    public:
        TParameterResource(Wt::WApplication* app, mo::IParam* param,
            const std::string& name = "default") :
            IParameterResource(app)
        {
            data = nullptr;
            data_param.SetUserDataPtr(&data);
            data_param.setName(name);
            data_param.setInput(param);
            data_param.SetMtx(&param->mtx());
            this->setParam(&data_param);
        }
        void handleParamUpdate(mo::Context* ctx, mo::IParam* param);

    private:
        const aq::SyncedMemory* data;
        mo::TypedInputParamPtr<aq::SyncedMemory> data_param;
    };
}
#endif
