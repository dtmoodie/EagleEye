#include "EagleLib/DataStream.hpp"
#include "EagleLib/Nodes/Node.h"

#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/archives/json.hpp>

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

IDataStream::Ptr IDataStream::Load(const std::string& config_file)
{
    rcc::shared_ptr<DataStream> stream = rcc::shared_ptr<DataStream>::Create();
    if (stream->LoadStream(config_file))
        return stream;
    return Ptr();
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
    StopThread();
    this->top_level_nodes.clear();
    if (!boost::filesystem::exists(filename))
    {
        LOG(warning) << "Stream config file doesn't exist: " << filename;
        return false;
    }
    std::string ext = boost::filesystem::extension(filename);
    if (ext == ".bin")
    {
        mo::StartSerialization();
        std::ifstream ifs(filename, std::ios::binary);
        cereal::BinaryInputArchive ar(ifs);
        ar(*this);
        mo::EndSerialization();
        return true;
    }
    else if (ext == ".json")
    {
        try
        {
            mo::StartSerialization();
            std::ifstream ifs(filename, std::ios::binary);
            cereal::JSONInputArchive ar(ifs);
            std::vector<rcc::shared_ptr<Nodes::Node>> all_nodes;
            ar.setNextName("nodes");
            ar.startNode();
            size_t size;
            ar(cereal::make_size_tag(size));
            all_nodes.resize(size);
            std::vector<std::vector<std::pair<std::string,std::string>>> inputs(size);
            for(int i = 0; i < size; ++i)
            {
                HandleNode(ar, all_nodes[i], inputs[i]);
            }
            for(auto& node : all_nodes)
            {
                node->SetDataStream(this);
            }
            std::vector<int> handled_node_indecies;
            for(int i = 0; i < inputs.size(); ++i)
            {
                if(inputs[i].size() == 0)
                {
                    handled_node_indecies.push_back(i);
                    AddNode(all_nodes[i]);
                }else
                {
                    bool connected = false;
                    for(int j = 0; j < inputs[i].size(); ++j)
                    {
                        std::string param_name = inputs[i][j].first;
                        std::string input_name = inputs[i][j].second;
                        if(input_name.size() == 0)
                        {
                            LOG(warning) << param_name << " input not set";
                            continue;
                        }
                        auto pos = input_name.find(':');
                        if(pos != std::string::npos)
                        {
                            std::string output_node_name = input_name.substr(0, pos);
                            auto node = GetNode(output_node_name);
                            if(!node)
                            {
                                LOG(warning) << "Unable to find node by name " << output_node_name;
                                continue;
                            }
                            auto output_param = node->GetOutput(input_name.substr(pos+1));
                            if(!output_param)
                            {
                                LOG(warning) << "Unable to find parameter " << input_name.substr(pos+1) << " in node " << node->GetTreeName();
                                continue;
                            }
                            auto input_param = all_nodes[i]->GetInput(param_name);
                            if(!input_param)
                            {
                                LOG(warning) << "Unable to find input parameter " << param_name << " in node " << all_nodes[i]->GetTreeName();
                                continue;
                            }
                            if(!all_nodes[i]->ConnectInput(node, output_param, input_param))
                            {
                                LOG(warning) << "Unable to connect " << output_param->GetTreeName() << " (" << output_param->GetTypeInfo().name() << ") to "
                                             << input_param->GetTreeName() << " (" << input_param->GetTypeInfo().name() << ")";
                            }else
                            {
                                connected = true;
                            }
                        }else
                        {
                            LOG(warning) << "Invalid input format " << input_name;
                        }
                    }
                    if(connected)
                    {
                       handled_node_indecies.push_back(i);
                    }
                }
            }
            for(int i = 0; i < all_nodes.size(); ++i)
            {
                if(std::find(handled_node_indecies.begin(), handled_node_indecies.end(), i) == handled_node_indecies.end())
                {
                    AddNode(all_nodes[i]);
                }
            }

            mo::EndSerialization();
        }catch(cereal::RapidJSONException&e)
        {
            LOG(warning) << "Unable to parse " << filename << " due to " << e.what();
            mo::EndSerialization();
            return false;
        }
        return true;
    }
    else if (ext == ".xml")
    {
        mo::StartSerialization();
        std::ifstream ifs(filename, std::ios::binary);
        cereal::XMLInputArchive ar(ifs);
        ar(*this);
        mo::EndSerialization();
        return true;
    }

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
