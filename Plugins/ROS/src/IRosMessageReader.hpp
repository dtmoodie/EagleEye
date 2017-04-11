#pragma once
#include <Aquila/Algorithm.h>
#include <MetaObject/IMetaObjectInfo.hpp>


namespace ros
{
class MessageReaderInfo: public mo::IMetaObjectInfo
{
public:
    virtual int CanHandleTopic(const std::string& type) const = 0;
    virtual void ListTopics(std::vector<std::string>& topics) const = 0;
};

class IMessageReader: public TInterface<ctcrc32("IMessageReader"), aq::Algorithm>
{
public:
    MO_BEGIN(IMessageReader)
        PARAM(std::string, subscribed_topic, "")
        //MO_SLOT(void, on_subscribed_topic_modified, mo::Context*, mo::IParameter*)
        PARAM_UPDATE_SLOT(subscribed_topic)
    MO_END;
    typedef MessageReaderInfo InterfaceInfo;
    static std::vector<std::string> ListSubscribableTopics();
    static rcc::shared_ptr<IMessageReader> Create(const std::string& topic);
    static int CanLoadTopic(const std::string& topic);
    virtual bool Subscribe(const std::string& topic) = 0;
};

} // namespace ros
