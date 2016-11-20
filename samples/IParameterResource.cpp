#ifdef HAVE_WT
#include "IParameterResource.hpp"
#include <Wt/WApplication>
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <cereal/archives/json.hpp>
#include <boost/thread/recursive_mutex.hpp>
using namespace vclick;

IParameterResource::IParameterResource(Wt::WApplication* app_):
    Wt::WStreamResource(),
    app(app_)
{
    param = nullptr;
    onParamUpdate = nullptr;
    ss = nullptr;
}
void IParameterResource::setParam(mo::IParameter* param_)
{
    std::lock_guard<std::mutex> lock(mtx);
    param = param_;
    if(param)
    {
        onParamUpdate = new mo::TypedSlot<void(mo::Context*, mo::IParameter*)>(std::bind(
            &IParameterResource::handleParamUpdate, this, std::placeholders::_1, std::placeholders::_2));
        this->connection = param->RegisterUpdateNotifier(onParamUpdate);
        ss = nullptr;
    }
}
IParameterResource::~IParameterResource()
{
    std::lock_guard<std::mutex> lock(mtx);
    delete onParamUpdate;
    if (ss)
    {
        delete ss;
        ss = nullptr;
    }
}
void IParameterResource::handleRequest(const Wt::Http::Request& request, Wt::Http::Response& response)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (ss)
    {
        handleRequestPiecewise(request, response, *ss);
    }
}

void IParameterResource::handleParamUpdate(mo::Context* ctx, mo::IParameter* param)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::stringstream* new_ss = new std::stringstream();
    if(this->param)
    {
        auto func = mo::SerializationFunctionRegistry::Instance()->
            GetJsonSerializationFunction(this->param->GetTypeInfo());
        if (func)
        {
            {
                cereal::JSONOutputArchive ar(*new_ss);
                boost::recursive_mutex::scoped_lock lock(this->param->mtx());
                func(this->param, ar);
            }
            std::iostream* old_ss;
            old_ss = ss;
            ss = new_ss;
            delete old_ss;
            //auto applock = app->getUpdateLock();
            //this->setChanged();
        }
    }
}
#endif
