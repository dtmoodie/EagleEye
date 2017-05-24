#ifdef HAVE_WT
#pragma once
#include <Wt/WStreamResource>
#include <MetaObject/Parameters/IParam.hpp>
#include <MetaObject/Signals/TypedSlot.hpp>
#include <sstream>
#include <mutex>

namespace vclick
{
    class IParameterResource : public Wt::WStreamResource
    {
    public:
        IParameterResource(Wt::WApplication* app);
        void setParam(mo::IParam* param);
        virtual ~IParameterResource();
        void handleRequest(const Wt::Http::Request& request, Wt::Http::Response& response);
        virtual void handleParamUpdate(mo::Context* ctx, mo::IParam* param);
    protected:
        mo::IParam* param;
        std::shared_ptr<mo::Connection> connection;
        mo::TypedSlot<void(mo::Context*, mo::IParam*)>* onParamUpdate;
        std::iostream* ss;
        std::mutex mtx;
        Wt::WApplication* app;
    };
}
#endif
