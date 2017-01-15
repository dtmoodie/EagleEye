#ifdef HAVE_WT
#pragma once
#include <Wt/WStreamResource>
#include <MetaObject/Parameters/IParameter.hpp>
#include <MetaObject/Signals/TypedSlot.hpp>
#include <sstream>
#include <mutex>

namespace vclick
{
    class IParameterResource : public Wt::WStreamResource
    {
    public:
        IParameterResource(Wt::WApplication* app);
        void setParam(mo::IParameter* param);
        virtual ~IParameterResource();
        void handleRequest(const Wt::Http::Request& request, Wt::Http::Response& response);
        virtual void handleParamUpdate(mo::Context* ctx, mo::IParameter* param);
    protected:
        mo::IParameter* param;
        std::shared_ptr<mo::Connection> connection;
        mo::TypedSlot<void(mo::Context*, mo::IParameter*)>* onParamUpdate;
        std::iostream* ss;
        std::mutex mtx;
        Wt::WApplication* app;
    };
}
#endif
