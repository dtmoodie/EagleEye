#pragma once

#include "CaffeNetHandler.hpp"
namespace mo
{
    template<class Type>
    struct MetaObjectInfoImpl<Type, EagleLib::Caffe::NetHandlerInfo>
            : public EagleLib::Caffe::NetHandlerInfo
    {
        std::vector<int> CanHandleNetwork(const caffe::Net<float>& net) const
        {
            return Type::CanHandleNetwork(net);
        }
    };

}
