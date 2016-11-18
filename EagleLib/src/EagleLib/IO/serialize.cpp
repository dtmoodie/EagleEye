#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include <EagleLib/IO/serialize.hpp>
#include <EagleLib/IO/memory.hpp>
#include <EagleLib/Nodes/Node.h>

#include "MetaObject/IO/Serializer.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/IO/Policy.hpp"
//#include "MetaObject/Parameters/IO/CerealPolicy.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;




bool EagleLib::Serialize(cereal::BinaryOutputArchive& ar, const Node* obj)
{
    if (auto func = mo::SerializerFactory::GetSerializationFunctionBinary(obj->GetTypeName()))
    {
        func(obj, ar);
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->GetParameters();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params)
        {
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetBinarySerializationFunction(param->GetTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->GetTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->GetTypeInfo().name();
            }
        }
    }
    return true;
}

bool EagleLib::DeSerialize(cereal::BinaryInputArchive& ar, Node* obj)
{
    return false;
}

bool EagleLib::Serialize(cereal::XMLOutputArchive& ar, const Node* obj)
{
    if (auto func = mo::SerializerFactory::GetSerializationFunctionXML(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->GetParameters();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params)
        {
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetXmlSerializationFunction(param->GetTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->GetTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->GetTypeInfo().name();
            }
        }
        return true;
    }
}

bool EagleLib::DeSerialize(cereal::XMLInputArchive& ar, Node* obj)
{
    return false;
}

bool EagleLib::Serialize(cereal::JSONOutputArchive& ar, const Node* obj)
{
    if (auto func = mo::SerializerFactory::GetSerializationFunctionJSON(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->GetParameters();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params)
        {
            if (param->CheckFlags(mo::Input_e))
            {
                mo::InputParameter* input = dynamic_cast<mo::InputParameter*>(param);
                if (input)
                {
                    auto input_source_param = input->GetInputParam();
                    if (input_source_param)
                    {
                        std::string input_source = input_source_param->GetTreeName();
                        std::string param_name = param->GetName();
                        ar(cereal::make_nvp(param_name, input_source));
                        continue;
                    }else
                    {
                        std::string blank;
                        std::string param_name = param->GetName();
                        ar(cereal::make_nvp(param_name, blank));
                        continue;
                    }
                }
            }
            if (param->CheckFlags(mo::Output_e))
                continue;
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetJsonSerializationFunction(param->GetTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->GetTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->GetTypeInfo().name();
            }
        }
        return true;
    }
}

bool EagleLib::DeSerialize(cereal::JSONInputArchive& ar, Node* obj)
{
    if (obj == nullptr)
        return false;
    if (auto func = mo::SerializerFactory::GetDeSerializationFunctionJSON(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->GetParameters();
        for (auto& param : params)
        {
            if (param->CheckFlags(mo::Input_e))
                continue;
            if (param->CheckFlags(mo::Output_e))
                continue;
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetJsonDeSerializationFunction(param->GetTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->GetTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->GetTypeInfo().name();
            }
            if (param->GetName() == "_dataStream")
            {
                auto typed = dynamic_cast<mo::ITypedParameter<rcc::weak_ptr<IDataStream>>*>(param);
                if (typed)
                {
                    obj->SetDataStream(typed->GetData().Get());
                }
            }
        }
        obj->SetParameterRoot(obj->GetTreeName());
        for (auto& param : params)
        {
            if (param->CheckFlags(mo::Input_e))
            {
                mo::InputParameter* input = dynamic_cast<mo::InputParameter*>(param);
                if (input)
                {
                    std::string input_source;
                    std::string param_name = param->GetName();
                    try
                    {
                        ar(cereal::make_nvp(param_name, input_source));
                    }
                    catch (cereal::Exception& e)
                    {
                        continue;
                    }
                    if (input_source.size())
                    {
                        auto token_index = input_source.find(':');
                        if (token_index != std::string::npos)
                        {
                            auto stream = obj->GetDataStream();
                            if(stream)
                            {
                                auto output_node = stream->GetNode(input_source.substr(0, token_index));
                                if (output_node)
                                {
                                    auto output_param = output_node->GetOutput(input_source.substr(token_index + 1));
                                    if (output_param)
                                    {
                                        //obj->ConnectInput(output_node, output_param, input, mo::BlockingStreamBuffer_e);
                                        obj->IMetaObject::ConnectInput(input, output_node, output_param, mo::BlockingStreamBuffer_e);
                                        obj->SetDataStream(output_node->GetDataStream());
                                        obj->SetContext(output_node->GetContext());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
}
