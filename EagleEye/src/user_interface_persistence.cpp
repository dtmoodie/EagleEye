#ifndef HAVE_OPENCV
#define HAVE_OPENCV
#endif

#include "user_interface_persistence.h"
#include <MetaObject/Parameters/IParameter.hpp>

void VariableStorage::LoadParams(UIPersistence* obj, const std::string& name)
{
    auto itr= loaded_parameters.find(name);
    if(itr != loaded_parameters.end())
    {
        auto params = obj->GetParameters();
        for(auto param : params)
        {
            auto itr2 = itr->second.find(param->GetName());
            if(itr2 != itr->second.end())
            {
                
            }
        }
    }
}
void VariableStorage::SaveParams(UIPersistence* obj, const std::string& name)
{
    /*auto& params = loaded_parameters[type.name()];
    auto all_params = This->getParameters();
    for(auto& param: all_params)
    {
        params[param->GetName()] = param->DeepCopy();
    }*/
}
void VariableStorage::SaveUI(const std::string& file_name)
{
    /*cv::FileStorage fs;
    fs.open(file_name, cv::FileStorage::WRITE);
    int index = 0;
    fs << "Count" << (int)loaded_parameters.size();
    for(auto itr: loaded_parameters)
    {
        fs << "Widget-" + boost::lexical_cast<std::string>(index) << "{";
        fs << "WidgetType" << itr.first;
        fs << "Parameters" << "{";
        for(auto itr2: itr.second)
        {
            Parameters::Persistence::cv::Serialize(&fs, itr2.second.get());
        }
        fs << "}"; // End parameters
        fs << "}"; // End widgets
        ++index;
    }*/
    
}
void VariableStorage::LoadUI(const std::string& file_name)
{
    /*try
    {
        cv::FileStorage fs;
        fs.open(file_name, cv::FileStorage::READ);
        int count = (int)fs["Count"];
        for(int i = 0; i < count; ++i)
        {
            auto widget_node = fs["Widget-" +boost::lexical_cast<std::string>(i)];
            auto type = (std::string)(widget_node)["WidgetType"];
            auto parameter_node = widget_node["Parameters"];
            auto& param_vec = loaded_parameters[type];
            for (auto itr2 = parameter_node.begin(); itr2 != parameter_node.end(); ++itr2)
            {
                auto node = *itr2;
                auto param = Parameters::Persistence::cv::DeSerialize(&node);
                if (param)
                {
                    param_vec[param->GetName()] = std::shared_ptr<mo::IParameter>(param);
                }
            }
        }
        
    }catch(...)
    {
    
    }*/
}

VariableStorage::VariableStorage()
{
    LoadUI();
}

VariableStorage::~VariableStorage()
{
    SaveUI();
}

VariableStorage* VariableStorage::Instance()
{
    static VariableStorage* inst = nullptr;
    if(inst == nullptr)
        inst = new VariableStorage();
    return inst;
}

UIPersistence::UIPersistence()
{
    //variable_storage::instance().load_parameters(this);
    VariableStorage::Instance()->LoadParams(this, "UIPersistence");
}


