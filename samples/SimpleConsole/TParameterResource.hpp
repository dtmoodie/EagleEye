#ifdef HAVE_WT
#pragma once
#include "IParameterResource.hpp"
#include <Aquila/types/SyncedMemory.hpp>
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

    template<> class TParameterResource<aq::SyncedMemory> : public IParameterResource
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
        const aq::SyncedMemory* data;
        mo::TypedInputParameterPtr<aq::SyncedMemory> data_param;
    };
}
#endif
