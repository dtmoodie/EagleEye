#pragma once

#include "CaffeNetHandler.hpp"
namespace mo
{
    template <class Type>
    struct MetaObjectInfoImpl<Type, aq::Caffe::NetHandlerInfo> : public aq::Caffe::NetHandlerInfo
    {
        std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net) const { return Type::CanHandleNetwork(net); }
    };
} // namespace mo
