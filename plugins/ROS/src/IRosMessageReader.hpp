#pragma once
#include "RosInterface.hpp"
#include "ros/master.h"
#include "ros/ros.h"
#include "ros/topic.h"
#include <Aquila/core/Algorithm.hpp>
#include <MetaObject/object/IMetaObjectInfo.hpp>

namespace ros
{
    class MessageReaderInfo : public mo::IMetaObjectInfo
    {
      public:
        virtual int CanHandleTopic(const std::string& type) const = 0;
        virtual void ListTopics(std::vector<std::string>& topics) const = 0;
    };

    class IMessageReader : public TInterface<IMessageReader, aq::Algorithm>
    {
      public:
        template <class T, int P>
        static int CanHandleTopic(const std::string& topic)
        {
            ros::master::V_TopicInfo ti;
            if (!ros::master::getTopics(ti))
                return 0;
            for (ros::master::V_TopicInfo::iterator it = ti.begin(); it != ti.end(); ++it)
            {
                if (it->name == topic)
                {
                    if (it->datatype == ros::message_traits::DataType<T>::value())
                    {
                        return P;
                    }
                }
            }
            return 0;
        }
        template <class T>
        static void ListTopics(std::vector<std::string>& topics)
        {
            aq::RosInterface::Instance();
            ros::master::V_TopicInfo ti;
            if (!ros::master::getTopics(ti))
                return;
            for (ros::master::V_TopicInfo::iterator it = ti.begin(); it != ti.end(); ++it)
            {
                if (it->datatype == ros::message_traits::DataType<T>::value())
                {
                    topics.push_back(it->name);
                }
            }
        }

        MO_BEGIN(IMessageReader)
        PARAM(std::string, subscribed_topic, "")
        PARAM_UPDATE_SLOT(subscribed_topic)
        MO_END;
        typedef MessageReaderInfo InterfaceInfo;
        static std::vector<std::string> ListSubscribableTopics();
        static rcc::shared_ptr<IMessageReader> create(const std::string& topic);
        static int CanLoadTopic(const std::string& topic);
        virtual bool subscribe(const std::string& topic) = 0;
    };

} // namespace ros
