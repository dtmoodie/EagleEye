#ifdef HAVE_WT
#include "TParameterResource.hpp"
#include <Wt/WApplication>
#include <MetaObject/serialization/SerializationFactory.hpp>
#include <cereal/archives/json.hpp>
using namespace vclick;

void TParameterResource<aq::SyncedMemory>::handleParamUpdate(mo::Context* ctx, mo::IParam* param)
{
    std::lock_guard<std::mutex> this_lock(mtx);
    std::stringstream* new_ss = new std::stringstream();
    auto func = mo::SerializationFactory::instance()->
        GetJsonSerializationFunction(this->param->getTypeInfo());
    boost::recursive_mutex::scoped_lock lock(this->param->mtx());
    dynamic_cast<mo::ITypedParameter<aq::SyncedMemory>*>(this->param)->GetDataPtr()->synchronize();
    if (func)
    {
        {
            cereal::JSONOutputArchive ar(*new_ss);
            func(this->param, ar);
        }
        std::iostream* old_ss;
        {
            
            old_ss = ss;
            ss = new_ss;
        }
        delete old_ss;
        auto applock = app->getUpdateLock();
        this->setChanged();
        
    }
}

void TParameterResourceRaw<cv::Mat>::handleParamUpdate(mo::Context* ctx, mo::IParam* param)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::stringstream* new_ss = new std::stringstream();
    new_ss->write((char*)data->data, data->size().area());
    if(ss)
        delete ss;
    ss = new_ss;
    auto applock = app->getUpdateLock();
    this->setChanged();
}
#endif
