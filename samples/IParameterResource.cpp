#ifdef HAVE_WT
#include "IParameterResource.hpp"
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <cereal/archives/json.hpp>

using namespace vclick;

IParameterResource::IParameterResource():
    Wt::WStreamResource()
{
    param = nullptr;
    onParamUpdate = nullptr;
    ss = nullptr;
}
void IParameterResource::setParam(mo::IParameter* param_)
{
    param = param_;
    onParamUpdate = new mo::TypedSlot<void(mo::Context*, mo::IParameter*)>(std::bind(
        &IParameterResource::handleParamUpdate, this, std::placeholders::_1, std::placeholders::_2));
    this->connection = param->RegisterUpdateNotifier(onParamUpdate);
    ss = nullptr;
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
    std::stringstream* new_ss = new std::stringstream();
    auto func = mo::SerializationFunctionRegistry::Instance()->
        GetJsonSerializationFunction(this->param->GetTypeInfo());
    if (func)
    {
        {
            cereal::JSONOutputArchive ar(*new_ss);
            func(this->param, ar);
        }
        std::stringstream* old_ss;
        {
            std::lock_guard<std::mutex> lock(mtx);
            old_ss = ss;
            ss = new_ss;
        }
        delete old_ss;
    }
}
#endif
