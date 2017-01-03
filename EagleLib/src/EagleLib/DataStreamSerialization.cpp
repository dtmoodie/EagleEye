#include "EagleLib/DataStream.hpp"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/ICoordinateManager.h"
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/archives/json.hpp>
#include "EagleLib/IO/JsonArchive.hpp"
#include "MetaObject/IO/Serializer.hpp"
#include "MetaObject/IO/Policy.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"

#include <EagleLib/IO/memory.hpp>
#include <boost/filesystem.hpp>
#include <fstream>

using namespace EagleLib;
template<typename AR> 
void DataStream::load(AR& ar)
{
    this->_load_parameters<AR>(ar, mo::_counter_<_DS_N_ - 1>());
    for(auto & node : top_level_nodes)
    {
        node->SetDataStream(this);
    }
    this->_load_parent<AR>(ar);
}

template<typename AR>
void DataStream::save(AR& ar) const
{
    ObjectId id = GetObjectId();
    std::string type = GetTypeName();
    ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
    ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
    ar(cereal::make_nvp("TypeName", type));
    this->_save_parameters<AR>(ar, mo::_counter_<_DS_N_ - 1>());
    this->_save_parent<AR>(ar);
}

IDataStream::Ptr IDataStream::Load(const std::string& config_file, const VariableMap& vm, const VariableMap& sm)
{
    rcc::shared_ptr<DataStream> stream_ = rcc::shared_ptr<DataStream>::Create();
    if(!stream_)
    {
    	LOG(error) << "Unable to create data stream";
    	return Ptr();
    }
    stream_->StopThread();
    stream_->top_level_nodes.clear();
    rcc::shared_ptr<IDataStream> stream(stream_);
    if (!boost::filesystem::exists(config_file))
    {
        LOG(warning) << "Stream config file doesn't exist: " << config_file;
        return Ptr();
    }
    std::string ext = boost::filesystem::extension(config_file);
    if (ext == ".bin")
    {
        std::ifstream ifs(config_file, std::ios::binary);
        cereal::BinaryInputArchive ar(ifs);
        //ar(stream);   
        return stream;
    }
    else if (ext == ".json")
    {
        try
        {
            std::ifstream ifs(config_file, std::ios::binary);
            EagleLib::JSONInputArchive ar(ifs, vm, sm);
            ar(stream);
        }
        catch (cereal::RapidJSONException&e)
        {
            LOG(warning) << "Unable to parse " << config_file << " due to " << e.what();
            return Ptr();
        }
        return stream;
    }
    else if (ext == ".xml")
    {
        std::ifstream ifs(config_file, std::ios::binary);
        cereal::XMLInputArchive ar(ifs);
        //ar(stream);
        return stream;
    }
    return Ptr();
}

void IDataStream::Save(const std::string& config_file, rcc::shared_ptr<IDataStream>& stream)
{
    stream->StopThread();
    if (boost::filesystem::exists(config_file))
    {
        LOG(info) << "Stream config file exists, overwiting: " << config_file;
    }
    std::string ext = boost::filesystem::extension(config_file);
    if (ext == ".json")
    {
        try
        {
            std::ofstream ofs(config_file, std::ios::binary);
            EagleLib::JSONOutputArchive ar(ofs);
            ar(stream);
        }
        catch (cereal::RapidJSONException&e)
        {
            LOG(warning) << "Unable to save " << config_file << " due to " << e.what();   
        }
    }
}

void HandleNode(cereal::JSONInputArchive& ar, rcc::shared_ptr<Nodes::Node>& node,
                std::vector<std::pair<std::string, std::string>>& inputs)
{
    ar.startNode();
    std::string name;
    std::string type;
    ar(CEREAL_NVP(name));
    ar(CEREAL_NVP(type));
    ar.startNode();
    node = mo::MetaObjectFactory::Instance()->Create(type.c_str());
    if(!node)
    {
        LOG(warning) << "Unable to create node with type: " << type;
        return;
    }
    node->SetTreeName(name);
    auto parameters = node->GetParameters();
    for(auto param : parameters)
    {
        if(param->CheckFlags(mo::Output_e) || param->CheckFlags(mo::Input_e))
            continue;
        auto func1 = mo::SerializationFunctionRegistry::Instance()->GetJsonDeSerializationFunction(param->GetTypeInfo());
        if (func1)
        {
            if(!func1(param, ar))
            {
                LOG(info) << "Unable to deserialize " << param->GetName();
            }
        }
    }

    ar.finishNode();
    try
    {
        ar(CEREAL_NVP(inputs));
    }catch(...)
    {
    
    }
    
    ar.finishNode();
}

bool DataStream::LoadStream(const std::string& filename)
{
    

    return false;
}


struct NodeSerializationInfo
{
    std::string name;
    std::string type;
    std::vector<mo::IParameter*> parameters;
    std::vector<std::pair<std::string, std::string>> inputs;
    void save(cereal::JSONOutputArchive& ar) const
    {
        ar(CEREAL_NVP(name));
        ar(CEREAL_NVP(type));
        ar.setNextName("parameters");
        ar.startNode();
        for(int i = 0; i < parameters.size(); ++i)
        {
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetJsonSerializationFunction(parameters[i]->GetTypeInfo());
            if (func1)
            {
                if (!func1(parameters[i], ar))
                {
                    LOG(debug) << "Unable to serialize " << parameters[i]->GetTreeName();
                }
            }
        }
        ar.finishNode();
        ar(CEREAL_NVP(inputs));
    }
    template<class AR> void save(AR& ar) const {}
    template<class AR> void load(AR& ar) {}
};

void PopulateSerializationInfo(Nodes::Node* node, std::vector<NodeSerializationInfo>& info)
{
    bool found = false;
    for(int i = 0; i < info.size(); ++i)
    {
        if(node->GetTreeName() == info[i].name)
            found = true;
    }
    if(!found)
    {
        NodeSerializationInfo node_info;
        node_info.name = node->GetTreeName();
        node_info.type = node->GetTypeName();
        auto all_params = node->GetParameters();
        for(auto& param : all_params)
        {
            if(param->GetName() == "_dataStream" ||
                    param->GetName() == "_children" ||
                    param->GetName() == "_parents" ||
                    param->GetName() == "_unique_id")
                continue;
            if(param->CheckFlags(mo::Control_e))
            {
                node_info.parameters.push_back(param);
            }
            if(param->CheckFlags(mo::Input_e))
            {
                std::string input_name;
                mo::InputParameter* input_param = dynamic_cast<mo::InputParameter*>(param);
                if(input_param)
                {
                    mo::IParameter* _input_param = input_param->GetInputParam();
                    if(_input_param)
                    {
                        input_name = _input_param->GetTreeName();
                    }
                }
                node_info.inputs.emplace_back(param->GetName(), input_name);
            }
        }
        info.push_back(node_info);
    }
    auto children = node->GetChildren();
    for(auto child : children)
    {
        PopulateSerializationInfo(child.Get(), info);
    }
}

bool DataStream::SaveStream(const std::string& filename)
{
    if (boost::filesystem::exists(filename))
    {
        LOG(warning) << "Overwriting existing stream config file: " << filename;
    }
    
    std::string ext = boost::filesystem::extension(filename);
    if (ext == ".bin")
    {
        mo::StartSerialization();
        std::ofstream ofs(filename, std::ios::binary);
        cereal::BinaryOutputArchive ar(ofs);
        ar(*this);
        mo::EndSerialization();
        return true;
    }
    else if (ext == ".json")
    {
        mo::StartSerialization();
        std::ofstream ofs(filename, std::ios::binary);
        cereal::JSONOutputArchive ar(ofs);
        std::vector<NodeSerializationInfo> serializationInfo;
        for(auto& node : top_level_nodes)
        {
            PopulateSerializationInfo(node.Get(), serializationInfo);
        }
        ar(cereal::make_nvp("nodes",serializationInfo));
        //ar(*this);
        mo::EndSerialization();
        return true;
    }
    else if (ext == ".xml")
    {
        mo::StartSerialization();
        std::ofstream ofs(filename, std::ios::binary);
        cereal::XMLOutputArchive ar(ofs);
        ar(*this);
        mo::EndSerialization();
        return true;
    }
    return false;
}
