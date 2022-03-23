#include "IRosMessageReader.hpp"
using namespace ros;

std::vector<std::string> IMessageReader::ListSubscribableTopics()
{
    std::vector<std::string> output;
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(IMessageReader::getHash());
    for (auto constructor : constructors)
    {
        if (auto info = dynamic_cast<const MessageReaderInfo*>(constructor->GetObjectInfo()))
        {
            info->ListTopics(output);
        }
    }
    return output;
}

int IMessageReader::CanLoadTopic(const std::string& topic)
{
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(IMessageReader::getHash());
    for (auto constructor : constructors)
    {
        if (auto info = dynamic_cast<const MessageReaderInfo*>(constructor->GetObjectInfo()))
        {
            int p = info->CanHandleTopic(topic);
            if (p > 0)
            {
                return p;
            }
        }
    }
    return 0;
}

void IMessageReader::on_subscribed_topic_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream*)
{
    this->subscribe(subscribed_topic);
}

rcc::shared_ptr<IMessageReader> IMessageReader::create(const std::string& topic)
{
    std::map<int, IObjectConstructor*> constructor_priority;
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(IMessageReader::getHash());
    for (auto constructor : constructors)
    {
        if (auto info = dynamic_cast<const MessageReaderInfo*>(constructor->GetObjectInfo()))
        {
            constructor_priority[info->CanHandleTopic(topic)] = constructor;
        }
    }
    if (constructor_priority.size())
    {
        if (constructor_priority.rbegin()->first > 0)
        {
            IObject* obj = constructor_priority.rbegin()->second->Construct();
            if (obj)
            {
                obj->Init(true);
                rcc::shared_ptr<IMessageReader> typed(*obj);
                if (typed)
                {
                    if (typed->subscribe(topic))
                    {
                        return typed;
                    }
                }
                else
                {
                    delete obj;
                }
            }
        }
    }
    return rcc::shared_ptr<IMessageReader>();
}
