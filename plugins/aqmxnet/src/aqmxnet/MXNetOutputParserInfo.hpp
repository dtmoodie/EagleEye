#pragma once
#include "MXNetOutputParser.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"

namespace mo
{

template <class Type>
struct MetaObjectInfoImpl<Type, aq::mxnet::MXNetOutputParser::MXNetOutputParserInfo>
    : public aq::mxnet::MXNetOutputParser::MXNetOutputParserInfo
{
    virtual int parserPriority(const ::mxnet::cpp::Symbol& sym) override { return Type::parserPriority(sym); }
};

} // namespace mo
