#pragma once

#include "IRosMessageReader.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"

namespace mo
{
    template <class Type>
    struct MetaObjectInfoImpl<Type, ros::MessageReaderInfo> : public ros::MessageReaderInfo
    {
        int CanHandleTopic(const std::string& type) const { return Type::CanHandleTopic(type); }
        void ListTopics(std::vector<std::string>& topics) const { Type::ListTopics(topics); }
    };
} // namespace mo
