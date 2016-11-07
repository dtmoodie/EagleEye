#include <EagleLib/IO/memory.hpp>
#include "MetaObject/IO/Serializer.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"

#include "MetaObject/IO/Policy.hpp"
#include <cereal/types/vector.hpp>
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"


using namespace EagleLib;
using namespace EagleLib::Nodes;


INSTANTIATE_META_PARAMETER(std::vector<rcc::shared_ptr<Node>>);

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
            {
                /*InputParameter* input = dynamic_cast<InputParameter*>(param);
                if (input)
                {
                auto input_source_param = input->GetInputParam();
                if (input_source_param)
                {
                std::string input_source = input_source_param->GetTreeName();
                std::string param_name = param->GetName();
                ar(cereal::make_nvp(param_name, input_source));
                }
                }*/
            }
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
        }
        return true;
    }
}
